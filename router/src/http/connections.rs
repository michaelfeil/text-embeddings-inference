//! Bounded HTTP connections, including idle clients, with graceful request drain.
use axum::Router;
use hyper_util::{
    rt::{TokioExecutor, TokioIo, TokioTimer},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use std::{
    future::Future,
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::Semaphore,
    task::JoinSet,
};

// Auto-protocol detection happens before Hyper's HTTP/1 header timer. Bound
// that phase as well, including an idle client or partial HTTP/2 preface.
struct PrefacedStream {
    stream: TcpStream,
    prefix: [u8; 24],
    length: usize,
    cursor: usize,
}
impl PrefacedStream {
    async fn read(mut stream: TcpStream) -> io::Result<Option<Self>> {
        let mut prefix = [0; 24];
        let mut length = 0;
        loop {
            let count = stream.read(&mut prefix[length..]).await?;
            if count == 0 {
                return Ok(None);
            }
            length += count;
            if length == prefix.len()
                || prefix[..length] != b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"[..length]
            {
                return Ok(Some(Self {
                    stream,
                    prefix,
                    length,
                    cursor: 0,
                }));
            }
        }
    }
}
impl AsyncRead for PrefacedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.cursor < self.length {
            let count = (self.length - self.cursor).min(buffer.remaining());
            buffer.put_slice(&self.prefix[self.cursor..self.cursor + count]);
            self.cursor += count;
            Poll::Ready(Ok(()))
        } else {
            Pin::new(&mut self.stream).poll_read(cx, buffer)
        }
    }
}
impl AsyncWrite for PrefacedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, data)
    }
    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write_vectored(cx, data)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

pub(crate) async fn serve(
    listener: TcpListener,
    app: Router,
    slots: Arc<Semaphore>,
    shutdown: impl Future<Output = ()>,
) {
    tokio::pin!(shutdown);
    let (drain, stopping) = tokio::sync::watch::channel(false);
    let mut connections = JoinSet::new();
    loop {
        // Reap completed tasks before accepting again so their bookkeeping
        // cannot accumulate across a steady stream of short connections.
        let permit = tokio::select! {
            biased;
            _ = &mut shutdown => break,
            Some(result) = connections.join_next(), if !connections.is_empty() => {
                if let Err(error) = result { tracing::error!("HTTP connection task failed: {error}"); }
                continue;
            }
            permit = slots.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let accepted = tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => accepted,
        };
        let (stream, _) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                tracing::warn!("HTTP accept failed: {error}");
                tokio::select! {
                    _ = &mut shutdown => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                }
                continue;
            }
        };
        let service = TowerToHyperService::new(app.clone());
        let mut stopping = stopping.clone();
        connections.spawn(async move {
            let _permit = permit;
            let stream = tokio::select! {
                _ = stopping.changed() => return,
                result = tokio::time::timeout(Duration::from_secs(10), PrefacedStream::read(stream)) => match result {
                    Ok(Ok(Some(stream))) => stream,
                    _ => return,
                },
            };
            let mut builder = Builder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(Duration::from_secs(10));
            builder.http2().max_concurrent_streams(128);
            let connection = builder.serve_connection(TokioIo::new(stream), service);
            tokio::pin!(connection);
            let result = tokio::select! {
                result = &mut connection => result,
                _ = stopping.changed() => {
                    connection.as_mut().graceful_shutdown();
                    connection.await
                }
            };
            if let Err(error) = result {
                tracing::debug!("HTTP connection ended: {error}");
            }
        });
    }
    // Stop accepting, close idle keep-alive connections, and allow active
    // requests to finish while retaining their connection permits.
    drop(listener);
    let _ = drain.send(true);
    while let Some(result) = connections.join_next().await {
        if let Err(error) = result {
            tracing::error!("HTTP connection task failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
    };

    #[tokio::test]
    async fn idle_connection_holds_capacity_and_disconnect_allows_next_client() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let app = Router::new().route("/", axum::routing::get(|| async { "healthy" }));
        let task = tokio::spawn(serve(listener, app, slots.clone(), async {
            let _ = stopped.await;
        }));
        let idle = TcpStream::connect(address).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while slots.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let mut next = TcpStream::connect(address).await.unwrap();
        next.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut buffer = [0; 1024];
        assert!(
            tokio::time::timeout(Duration::from_millis(50), next.read(&mut buffer))
                .await
                .is_err()
        );
        drop(idle);
        let count = tokio::time::timeout(Duration::from_secs(2), next.read(&mut buffer))
            .await
            .unwrap()
            .unwrap();
        assert!(std::str::from_utf8(&buffer[..count])
            .unwrap()
            .contains("200 OK"));
        drop(next);
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .build()
            .unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            client.get(format!("http://{address}/")).send(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.version(), reqwest::Version::HTTP_2);
        assert_eq!(response.text().await.unwrap(), "healthy");
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn shutdown_drains_active_request_before_releasing_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let started = Arc::new(tokio::sync::Notify::new());
        let finish = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route(
            "/",
            axum::routing::get({
                let started = started.clone();
                let finish = finish.clone();
                move || {
                    let started = started.clone();
                    let finish = finish.clone();
                    async move {
                        started.notify_one();
                        finish.notified().await;
                        "completed"
                    }
                }
            }),
        );
        let task = tokio::spawn(serve(listener, app, slots.clone(), async {
            let _ = stopped.await;
        }));
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        stop.send(()).unwrap();
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        assert_eq!(slots.available_permits(), 0);
        finish.notify_one();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(std::str::from_utf8(&response)
            .unwrap()
            .contains("completed"));
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(slots.available_permits(), 1);
    }
}
