# Licensing and proposed relicensing

This project is a fork of [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).
The upstream Apache-2.0 license remains in [LICENSE](LICENSE); a copy is also
available at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Existing copyright
and attribution notices remain in the source files and documentation.

No replacement license has been selected. “TBD - fork of what was originally
huggingface/text-embeddings-inference” records a pending decision about fork
modifications; it is not license text and does not amend existing license grants.
Third-party components retain their own terms, indexed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## INFOIL proposal (not adopted)

The proposed [Infinity Optimized Inference License, INFOIL 1.0](LICENSES/INFOIL-1.0-DRAFT.txt)
is available for review. It is a custom source-available proposal, not an
open-source license. Adding this draft does not adopt it, change `LICENSE`,
or restrict any rights already granted under Apache-2.0 or another license.

The draft covers commercial inference providers and deployments on their
capacity, including Fireworks AI, Modal, Together AI, CoreWeave, and Nebius.
The general definition is broader than those names: helping even one paying
customer stand up an inference endpoint on supplied capacity qualifies,
including when billing is for GPU hours rather than tokens. The restriction
also covers internal, research, testing, and free deployments on restricted
capacity. It does not depend on requests arriving through OpenRouter.

Baseten deployments are expressly permitted, including infrastructure Baseten
leases from other providers while actually operating the model-serving
deployment. A customer that is itself an inference provider may use that
exception within a Baseten deployment. An endpoint operated independently
elsewhere does not qualify merely because traffic or billing passes through
Baseten. Naming Baseten as a permitted operator does not identify it as the
copyright owner or Licensor.

Distributing covered modifications would require retaining INFOIL and giving
recipients the corresponding source. Private modifications and network use
alone would not trigger publication. Existing Apache and other third-party
rights remain independently available. There is no blanket contributor
relicensing grant; separate commercial exceptions require authority over
the affected rights.

Examples below describe the proposed terms only. They do not change the
license of the current repository.

| Scenario | Proposed INFOIL treatment |
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

Adoption remains a separate change. Its notice must identify the authorized
Licensor, covered code and rights, first covered full commit or release,
effective date, and third-party or previously licensed exclusions. None of
those adoption facts is established merely by publishing this proposal.

Drafting references: [historical HFOIL 1.0](https://github.com/huggingface/text-generation-inference/blob/v1.0.0/LICENSE),
[Apache-2.0 sections 2 and 4](https://www.apache.org/licenses/LICENSE-2.0), and
[Baseten's published legal name](https://www.baseten.co/terms-and-conditions/).
The INFOIL draft uses its own definitions and terms; it is not a claim of
endorsement by Hugging Face, Baseten, or any named provider.

## File structure

| File | Purpose |
| --- | --- |
| `LICENSE` | Current upstream license; if new terms are adopted, clearly explain their scope here and include or reference their full text. |
| `LICENSES/Apache-2.0.txt` | Retained copy of the upstream license. |
| `LICENSES/INFOIL-1.0-DRAFT.txt` | Proposed INFOIL terms; not an effective license change. |
| `NOTICE` | Attribution for the upstream project and the fork. |
| `THIRD_PARTY_NOTICES.md` | Index of bundled component notices and dependency audit scope. |
| `README.md` | User-facing origin acknowledgment and license summary. |

## Before adopting new terms

1. Identify the copyright holders and existing license grants for fork
   contributions. Obtain authorization where the proposed change requires it;
   do not assume every contribution is owned by the fork maintainer.
2. Choose the actual license and specify whether it covers fork modifications
   or the derivative distribution as a whole. Review compatibility with the
   inherited code and the dependencies included in each release artifact.
3. Put the full chosen terms in the repository, define their scope in `LICENSE`,
   and update the README, package metadata, and API metadata consistently.
   Do not replace inherited file notices or label third-party code as owned by
   the fork. Modified upstream files need prominent notices of modification.
4. Include applicable license texts and attribution notices in source archives,
   binary releases, and container images. Audit each artifact's included
   dependencies; the current third-party index is not a complete inventory of
   transitive dependencies or container contents.
5. Record the effective release or commit for the new terms. Do not describe
   this as revoking rights already granted under an earlier license.

Apache-2.0 permits different terms for modifications or derivative works as a
whole, subject to its redistribution conditions, including retaining applicable
notices and providing a copy of the Apache license. See
[Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0) and the
[Apache licensing FAQ](https://www.apache.org/foundation/license-faq.html).
