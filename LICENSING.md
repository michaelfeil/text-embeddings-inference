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

## File structure

| File | Purpose |
| --- | --- |
| `LICENSE` | Current upstream license; if new terms are adopted, clearly explain their scope here and include or reference their full text. |
| `LICENSES/Apache-2.0.txt` | Retained copy of the upstream license. |
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
