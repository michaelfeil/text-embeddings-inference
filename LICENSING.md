# infinity_v3 licensing

This fork adopts [INFOIL 1.0](LICENSES/INFOIL-1.0.txt). The authoritative
[Adoption Notice and complete license bundle](LICENSE) identify the scope,
licensors, effective date, and retained independent licenses. INFOIL is a
custom source-available license with use restrictions; it is not an
unrestricted open-source license.

## Proposed successor: platform blacklist for commercial use (not adopted)

The [Token Service Elastic License draft](docs/licensing/TSEL-1.0-DRAFT.txt)
proposes a simple rule: **you may not deploy Infinity for commercial use on
the listed platforms**, whether you are the platform operator or its customer.
It is a custom license, not Elastic License 2.0. INFOIL 1.0 remains in force;
this proposal does not relicense existing contributions or replace its grants.

The [OpenRouter snapshot](docs/licensing/openrouter-providers-2026-10-13.json)
records 93 public-directory entries retrieved on 2026-10-13, of which 89 are
explicitly listed in the draft. Azure, Baseten, Google Vertex,
and Google AI Studio are excluded under the approved-provider exceptions;
Oracle Cloud is also expressly permitted but was absent from the directory.
General AWS infrastructure such as EC2 is permitted; **Amazon Bedrock is
explicitly prohibited for commercial deployment**. A listed service does not
become permitted merely because it runs on AWS or another allowed cloud.
The API returned 19 additional entries not shown in the public directory,
including a test-like `FakeProvider`; those entries are recorded for comparison
and do not automatically enter the restriction list. OpenRouter changes do
not alter the fixed list attached to this proposed version.

The proposal allows unlisted platforms and private hardware, without revenue
tests. It preserves noncommercial research on listed capacity and targets
employees only when acting for a listed platform's operator toward a prohibited
deployment. Its rewrite condition covers manual work, coding agents, and
contractors or subprocessors using covered materials or their implementation
design to prepare a replacement for a listed platform's commercial service
or another prohibited deployment, including ports to C or Go.

| Scenario | Proposed treatment, if adopted |
| --- | --- |
| Commercial use on private hardware or an unlisted provider | Permitted. |
| Commercial Infinity deployment directly on AWS EC2 | Permitted. |
| Commercial Infinity deployment within or as part of Amazon Bedrock | Restricted; AWS infrastructure permission does not cover Bedrock. |
| Noncommercial research on Modal or Together capacity | Permitted. |
| Commercial endpoint on a listed provider, including private GPU-hour billing | Restricted. |
| Ordinary company deploying its own private application on Modal or Together | Restricted for commercial use, even if it does not sell inference. |
| Independent personal research by a listed provider's employee | Permitted; employment alone is not a ban. |
| Employee or contractor adapting the software for a listed provider's prohibited deployment | Restricted, including before deployment. |
| Manual, agent, or subprocessor rewrite using the covered design to prepare a competing listed-provider service | Restricted, even if rewritten in C or Go. |
| NVIDIA hardware, Hugging Face model downloads, or Cloudflare DNS without execution on a listed service | Not restricted merely because of the supplier. |
| Anthropic-operated Claude Platform on AWS | Listed; using AWS underneath does not make Anthropic's service an AWS exception. |
| Baseten actually operates its own deployment on capacity leased from a listed platform's operator | Permitted under the infrastructure exception; running inside the listed platform's service is still restricted. |

The rewrite condition is a proposed restriction on use of licensed material,
not a claim that copyright covers every similar design. Copyright does not
protect general ideas, program logic, or algorithms, and independent grants
and nonwaivable rights remain available. See the
[U.S. Copyright Office's computer-program guidance](https://www.copyright.gov/register/tx-programs.html).
The draft also preserves the existing same-license distribution and source
requirements. Adoption requires a new scope notice and permission from the
applicable rights holders; repository push access alone does not supply it.

## Adoption and existing rights

- **Licensor:** Infinity Core Contributors, meaning each copyright holder or
  authorized licensor for their own contributions, identified by source
  notices and contribution records. This collective label does not assign
  copyright, create a legal entity, or authorize relicensing others' work.
- **Effective date:** 2026-10-12.
- **First covered source release:** `infinity_v3-infoil-1.0`, identified by
  the Git tag of the same name. This marks the license transition, not a
  public Docker release.
- **Pre-adoption base:** `9486dd9ee89575442fda1a491c381046b37e116e`.

INFOIL covers the original fork contributions and copyrightable modifications
first offered under this adoption, and subsequent contributions accepted on
that basis, within the scope in `LICENSE`. It also governs the contributors'
covered rights in the derivative distribution as a whole. It does not
revoke rights already granted for inherited or previously published code.
Earlier Apache-covered source can still be used and redistributed under its
existing grant, including by people who obtain a copy after this transition.

Hugging Face Text Embeddings Inference retains Apache-2.0. The complete prior
root license and upstream attribution remain in the root `LICENSE` bundle;
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) remains unchanged.
Bundled components keep their own terms and source notices, indexed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Naming a permitted operator
or a maintainer does not establish ownership of those components.

## Provider restrictions and the Baseten exception

Commercial inference providers and deployments on their capacity are
restricted, including Fireworks AI, Modal, Together AI, CoreWeave, and Nebius.
The general definition is broader than those names: helping even one paying
customer stand up an inference endpoint on supplied capacity qualifies,
including when billing is for GPU hours rather than tokens. Internal,
research, testing, and free deployments on restricted capacity are also
covered. Routing through OpenRouter or another intermediary does not avoid
the restrictions. See INFOIL Sections 1 and 3 for the full definitions.

Section 4 expressly permits Baseten-operated deployments, including capacity
Baseten leases from other providers while actually operating the serving
deployment. A customer that is itself an inference provider may use that
exception within a Baseten deployment. An independently operated endpoint
elsewhere does not qualify merely because traffic or billing passes through
Baseten. This exception does not name Baseten as the owner or Licensor of
other contributors' work.

| Scenario | INFOIL treatment |
| --- | --- |
| Internal application on company-owned hardware, where the company is not an inference provider | Permitted. |
| Commercial application on unrestricted capacity that does not expose a general-purpose inference or model-deployment service | Permitted. |
| One private customer endpoint set up by a provider on its rented GPUs, billed by GPU-hour | Restricted; public access and per-token billing are unnecessary. |
| Free research deployment on Modal, CoreWeave, or another restricted provider | Restricted. |
| A restricted provider's internal benchmark on its own machines | Restricted. |
| A customer-operated deployment on another cloud that itself qualifies as a commercial inference provider | Restricted, even if the customer rents only generic capacity. |
| Baseten-operated endpoint, including one using capacity leased from a restricted provider | Permitted under the Baseten exception. |
| An inference provider reselling results served entirely by a Baseten deployment | Permitted within that deployment under the Baseten exception. |
| A customer operates an endpoint elsewhere and routes requests through Baseten | No Baseten exception. |
| An allowed application calls an unrelated external inference API | That call alone does not make its execution infrastructure restricted. |
| A recipient uses or redistributes an earlier Apache-licensed version | Governed by the earlier grant, which INFOIL does not revoke. |

## Contributions and redistribution

New covered contributions are offered under INFOIL 1.0 unless different terms
are expressly agreed before inclusion. Contributors retain ownership.
There is no blanket grant to relicense their work or issue commercial
exceptions on their behalf; a separate agreement must come from parties
holding the needed rights. The Baseten exception is part of INFOIL itself.

Redistributed covered modifications must retain this version of INFOIL and
include corresponding source, as specified in Section 5. Private
modifications and network use alone do not require publication. Preserve
all inherited notices, prominently identify modified files, and distinguish
new INFOIL-covered modifications from independently licensed material.

The root `LICENSE` contains the Adoption Notice and full INFOIL and retained
Apache texts so packages and native library installations carry the actual
terms rather than only links. First-party workspace packages reference this
bundle through `license-file`; independently licensed extension crates keep
their own declarations.

Docker builds install that bundle, the licensing notes, and bundled license
files at `/usr/share/licenses/infinity_v3/`. The collection is not a complete
audit of transitive dependencies, model weights, CUDA libraries, or base
images; distributors must retain the additional notices their artifacts
require. LibTorch's existing native notices remain installed with its library.

## License files

| File | Purpose |
| --- | --- |
| `LICENSE` | Adoption Notice, scope and exclusions, complete INFOIL terms, and retained original Apache license. |
| `LICENSES/INFOIL-1.0.txt` | INFOIL 1.0 terms. |
| `LICENSES/Apache-2.0.txt` | Retained upstream license copy. |
| `NOTICE` | Upstream and fork attribution; does not amend license grants. |
| `THIRD_PARTY_NOTICES.md` | Bundled component notice index and dependency-audit scope. |

References: [historical HFOIL 1.0](https://github.com/huggingface/text-generation-inference/blob/v1.0.0/LICENSE),
[Apache-2.0 sections 2 and 4](https://www.apache.org/licenses/LICENSE-2.0), and
[Baseten's published legal name](https://www.baseten.co/terms-and-conditions/).
INFOIL is a distinct license; these references do not imply endorsement.
