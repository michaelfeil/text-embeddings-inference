//! Typed response serialization and Accept negotiation shared by every HTTP route.
use axum::{
    body::{to_bytes, Body},
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::{de::DeserializeOwned, Serialize};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Native,
    Json,
    MessagePack,
    LegacyMessagePack,
}

impl Format {
    fn media_type(self) -> &'static str {
        match self {
            Self::Native => unreachable!("native response already has a media type"),
            Self::Json => "application/json",
            Self::MessagePack => "application/msgpack",
            Self::LegacyMessagePack => "application/x-msgpack",
        }
    }
}

tokio::task_local! { static FORMAT: Format; }

/// JSON request extractor with directly serialized JSON or named-map MessagePack responses.
/// Request parsing and rejection types remain Axum's, and OpenAPI retains the original T.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Json<T>(pub T);

#[axum::async_trait]
impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for Json<T> {
    type Rejection = JsonRejection;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        axum::Json::<T>::from_request(request, state)
            .await
            .map(|axum::Json(value)| Self(value))
    }
}

impl<T: Serialize> IntoResponse for Json<T> {
    fn into_response(self) -> Response {
        let format = FORMAT.try_with(|format| *format).unwrap_or(Format::Json);
        match format {
            Format::Native | Format::Json => axum::Json(self.0).into_response(),
            Format::MessagePack | Format::LegacyMessagePack => {
                match rmp_serde::to_vec_named(&self.0) {
                    Ok(bytes) => {
                        ([(header::CONTENT_TYPE, format.media_type())], bytes).into_response()
                    }
                    Err(error) => {
                        tracing::error!("MessagePack response serialization failed: {error}");
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Response serialization failed",
                        )
                            .into_response()
                    }
                }
            }
        }
    }
}

// Check permission only: positive q values do not rank formats. A specific
// q=0 exclusion overrides a wildcard that would otherwise allow the media type.
fn allows(headers: &HeaderMap, media_type: &str) -> bool {
    let mut matched = (0, false);
    for value in headers.get_all(header::ACCEPT) {
        let Ok(value) = value.to_str() else { continue };
        for range in value.split(',') {
            let mut parts = range.split(';');
            let range = parts.next().unwrap_or_default().trim();
            let specificity = if range.eq_ignore_ascii_case(media_type) {
                3
            } else if range == "*/*" {
                1
            } else if range.ends_with("/*")
                && media_type
                    .split('/')
                    .next()
                    .is_some_and(|kind| range[..range.len() - 2].eq_ignore_ascii_case(kind))
            {
                2
            } else {
                continue;
            };
            let mut allowed = true;
            for parameter in parts {
                if let Some((name, value)) = parameter.trim().split_once('=') {
                    if name.trim().eq_ignore_ascii_case("q") {
                        allowed = value
                            .trim()
                            .parse::<f32>()
                            .is_ok_and(|q| q > 0.0 && q <= 1.0);
                    }
                }
            }
            matched = matched.max((specificity, allowed));
        }
    }
    matched.1
}

fn select(headers: &HeaderMap, native: Option<&str>) -> Option<Format> {
    if !headers.contains_key(header::ACCEPT) {
        return Some(Format::Native);
    }
    // MessagePack wins whenever allowed, regardless of header order or weights.
    for (format, media_type) in [
        (Format::MessagePack, "application/msgpack"),
        (Format::LegacyMessagePack, "application/x-msgpack"),
    ] {
        if allows(headers, media_type) {
            return Some(format);
        }
    }
    if native.is_some_and(|mime| allows(headers, mime)) {
        Some(Format::Native)
    } else if allows(headers, "application/json") {
        Some(Format::Json)
    } else {
        None
    }
}

pub(crate) async fn negotiate(request: Request, next: Next) -> Response {
    let headers = request.headers().clone();
    let initial = select(&headers, None);
    // Native text routes can satisfy Accept values outside the API formats.
    let native_route =
        request.uri().path().starts_with("/docs") || request.uri().path() == "/metrics";
    if initial.is_none() && !native_route {
        let mut response = StatusCode::NOT_ACCEPTABLE.into_response();
        response
            .headers_mut()
            .append(header::VARY, HeaderValue::from_static("Accept"));
        return response;
    }
    let head = request.method() == Method::HEAD;
    let response = FORMAT
        .scope(initial.unwrap_or(Format::Native), next.run(request))
        .await;
    let (mut parts, body) = response.into_parts();
    parts
        .headers
        .append(header::VARY, HeaderValue::from_static("Accept"));
    if parts.status == StatusCode::NO_CONTENT
        || parts.status == StatusCode::NOT_MODIFIED
        || parts.status.is_informational()
    {
        return Response::from_parts(parts, body);
    }
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    let native = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    let Some(format) = select(&headers, native) else {
        parts.status = StatusCode::NOT_ACCEPTABLE;
        parts.headers.remove(header::CONTENT_TYPE);
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::empty());
    };
    if format == Format::Native || native == Some(format.media_type()) {
        return Response::from_parts(parts, body);
    }
    let original_json = native == Some("application/json");
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.media_type()),
    );
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::ETAG);
    if head {
        return Response::from_parts(parts, Body::empty());
    }
    // Only untyped text/empty/framework responses need buffering. Typed API responses
    // are serialized directly above, preserving f32 and integer types without JSON parsing.
    let encoded = async {
        let bytes = to_bytes(body, usize::MAX).await.map_err(|_| ())?;
        let value = if original_json {
            serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|_| ())?
        } else if bytes.is_empty() {
            serde_json::Value::Null
        } else if let Ok(text) = std::str::from_utf8(&bytes) {
            serde_json::Value::String(text.to_owned())
        } else {
            serde_json::to_value(bytes.to_vec()).map_err(|_| ())?
        };
        if format == Format::Json {
            serde_json::to_vec(&value).map_err(|_| ())
        } else {
            rmp_serde::to_vec_named(&value).map_err(|_| ())
        }
    }
    .await;
    match encoded {
        Ok(bytes) => Response::from_parts(parts, Body::from(bytes)),
        Err(()) => {
            parts.status = StatusCode::INTERNAL_SERVER_ERROR;
            Response::from_parts(parts, Body::empty())
        }
    }
}

pub(crate) fn document(doc: &mut utoipa::openapi::OpenApi) {
    for (name, path) in &mut doc.paths.paths {
        for operation in path.operations.values_mut() {
            for response in operation.responses.responses.values_mut() {
                let utoipa::openapi::RefOr::T(response) = response else {
                    continue;
                };
                // Empty success responses are still bodyless by default, but encode
                // null when a JSON or MessagePack representation is requested.
                if name == "/health" && response.content.is_empty() {
                    let schema = utoipa::openapi::schema::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::schema::SchemaType::Value)
                        .nullable(true)
                        .enum_values(Some([serde_json::Value::Null]))
                        .build();
                    response.content.insert(
                        "application/json".into(),
                        utoipa::openapi::ContentBuilder::new()
                            .schema(schema)
                            .build(),
                    );
                }
                if !response.content.contains_key("application/json") {
                    if let Some(content) = response.content.get("text/plain").cloned() {
                        response.content.insert("application/json".into(), content);
                    }
                }
                if let Some(content) = response
                    .content
                    .get("application/json")
                    .or_else(|| response.content.get("text/plain"))
                {
                    let content = utoipa::openapi::ContentBuilder::new()
                        .schema(content.schema.clone())
                        .build();
                    response
                        .content
                        .insert("application/msgpack".into(), content);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        routing::{get, post},
        Router,
    };
    use serde::Deserialize;
    use tower::ServiceExt;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Payload {
        integer: u64,
        score: f32,
    }

    fn app() -> Router {
        Router::new()
            .route(
                "/info",
                get(|| async {
                    tokio::task::yield_now().await;
                    Json(Payload {
                        integer: u64::MAX,
                        score: 0.5,
                    })
                }),
            )
            .route(
                "/echo",
                post(|Json(value): Json<Payload>| async move { Json(value) }),
            )
            .route(
                "/error",
                get(|| async {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error":"bad input"})),
                    )
                }),
            )
            .route("/metrics", get(|| async { "requests 1\n" }))
            .route(
                "/docs",
                get(|| async { axum::response::Html("<html>docs</html>") }),
            )
            .route("/health", get(|| async {}))
            .route("/empty", get(|| async { StatusCode::NO_CONTENT }))
            .layer(axum::middleware::from_fn(negotiate))
    }

    async fn call(path: &str, accept: Option<&str>) -> Response {
        let mut request = Request::builder().uri(path);
        if let Some(accept) = accept {
            request = request.header(header::ACCEPT, accept);
        }
        app()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[test]
    fn messagepack_wins_when_allowed() {
        for (accept, expected) in [
            ("*/*", Some(Format::MessagePack)),
            ("application/*", Some(Format::MessagePack)),
            (
                "application/json,application/msgpack;q=0.1",
                Some(Format::MessagePack),
            ),
            ("APPLICATION/X-MSGPACK", Some(Format::LegacyMessagePack)),
            (
                "application/msgpack;q=0,application/x-msgpack;q=0,application/*",
                Some(Format::Json),
            ),
            ("application/xml", None),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT, accept.parse().unwrap());
            assert_eq!(select(&headers, None), expected, "{accept}");
        }
    }

    #[tokio::test]
    async fn typed_responses_preserve_numeric_types_and_request_isolation() {
        let (json, msgpack, alias) = tokio::join!(
            call("/info", None),
            call("/info", Some("application/json,application/msgpack;q=0.1")),
            call("/info", Some("application/x-msgpack"))
        );
        assert_eq!(json.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(
            alias.headers()[header::CONTENT_TYPE],
            "application/x-msgpack"
        );
        assert_eq!(
            msgpack.headers()[header::CONTENT_TYPE],
            "application/msgpack"
        );
        assert_eq!(msgpack.headers()[header::VARY], "Accept");
        let bytes = to_bytes(msgpack.into_body(), 1024).await.unwrap();
        assert_eq!(bytes[0], 0x82); // Named map, not a positional struct array.
        assert!(bytes.ends_with(&[0xca, 0x3f, 0, 0, 0])); // Native f32, without JSON/f64 conversion.
        let decoded = rmp_serde::from_slice::<Payload>(&bytes).unwrap();
        assert_eq!(
            decoded,
            Payload {
                integer: u64::MAX,
                score: 0.5
            }
        );
        assert_eq!(
            serde_json::from_slice::<Payload>(&to_bytes(json.into_body(), 1024).await.unwrap())
                .unwrap(),
            decoded
        );
    }

    #[tokio::test]
    async fn errors_text_and_bodyless_responses() {
        let error = call("/error", Some("application/msgpack")).await;
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            rmp_serde::from_slice::<serde_json::Value>(
                &to_bytes(error.into_body(), 1024).await.unwrap()
            )
            .unwrap()["error"],
            "bad input"
        );
        let malformed = Request::builder()
            .method(Method::POST)
            .uri("/echo")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/msgpack")
            .body(Body::from("bad JSON"))
            .unwrap();
        let error = app().oneshot(malformed).await.unwrap();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert!(
            rmp_serde::from_slice::<String>(&to_bytes(error.into_body(), 1024).await.unwrap())
                .unwrap()
                .contains("Failed to parse")
        );

        let native = call("/metrics", None).await;
        assert_eq!(
            to_bytes(native.into_body(), 1024).await.unwrap(),
            "requests 1\n"
        );
        let encoded = call("/metrics", Some("*/*")).await;
        assert_eq!(
            rmp_serde::from_slice::<String>(&to_bytes(encoded.into_body(), 1024).await.unwrap())
                .unwrap(),
            "requests 1\n"
        );
        assert!(
            call("/docs", Some("text/html")).await.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        assert_eq!(
            call("/docs", Some("text/html,*/*;q=0.1")).await.headers()[header::CONTENT_TYPE],
            "application/msgpack"
        );
        let health = call("/health", Some("application/msgpack")).await;
        assert_eq!(
            to_bytes(health.into_body(), 1024).await.unwrap().as_ref(),
            &[0xc0]
        );
        let empty = call("/empty", Some("application/msgpack")).await;
        assert_eq!(empty.status(), StatusCode::NO_CONTENT);
        assert!(to_bytes(empty.into_body(), 1024).await.unwrap().is_empty());
        assert_eq!(
            call("/info", Some("application/xml")).await.status(),
            StatusCode::NOT_ACCEPTABLE
        );
        let head = Request::builder()
            .method(Method::HEAD)
            .uri("/health")
            .header(header::ACCEPT, "application/msgpack")
            .body(Body::empty())
            .unwrap();
        let response = app().oneshot(head).await.unwrap();
        assert!(to_bytes(response.into_body(), 1024)
            .await
            .unwrap()
            .is_empty());
    }
}
