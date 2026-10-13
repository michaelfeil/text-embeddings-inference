# infinity_v3 licensing

Covered fork contributions are offered under a choice of:

- [Infinity License 1.0 for Commercial Use](LICENSES/Infinity-License-1.0-for-Commercial-Use.txt).
- [Infinity License 1.0 for Noncommercial Use](LICENSES/Infinity-License-1.0-for-Noncommercial-Use.txt).

The [Adoption Notice and complete license bundle](LICENSE) define the covered
rights, exclusions, and effective version. Adoption takes effect at the merge
that first introduces that notice into `mf/faster-tei`; that merge commit and
date identify the first covered source version and effective date. The
pre-adoption integration base is `ec11d07f4a0c354279366c616ac9184cc0c84850`.

These are custom source-available licenses. **Michael Feil** is the author of
the license texts. **Infinity Core Contributors** describes each rights
holder or authorized licensor for their own contributions; it does not
assign ownership or permit one contributor to relicense another's work.

## Choose the alternative that permits your use

Commercial enterprise use is allowed on your own hardware or capacity
supplied by an unlisted provider, subject to the commercial platform
restrictions. Noncommercial use, including research, education, and personal
experimentation, is allowed on every platform. Business operations and
product or service development are commercial even before revenue.

This is a choice between the two alternatives, not a requirement to satisfy
both sets of use restrictions. Each use must comply with the applicable
alternative; a noncommercial deployment does not authorize a later
commercial deployment. Distributions must preserve both alternatives so
recipients and users of covered modifications retain the same choice.

| Use | License treatment for covered rights |
| --- | --- |
| Enterprise application on its own hardware or an unlisted provider's capacity | Allowed under the commercial license, subject to its platform restrictions. |
| Commercial deployment on a listed platform, including private endpoints, rented GPUs, or one customer's endpoint | Prohibited for operators and customers regardless of billing unit. |
| Ordinary AWS infrastructure such as EC2 | Allowed under the commercial license; Amazon Bedrock is separately prohibited. |
| Azure, Baseten, Google Cloud / AI Studio, or Oracle Cloud | Expressly permitted by the commercial license. |
| Noncommercial research, education, or personal use on Modal, Together, Fireworks, or another listed platform | Allowed under the noncommercial license. |
| Academic research paid for by grants, researcher salaries, or generic compute rental | Those payments alone do not make the research commercial. |
| A provider sells an Infinity-based inference service to one research customer | Commercial; the customer's research purpose does not exempt the provider. |
| Employee's independent noncommercial personal project | Allowed; employment alone does not restrict it. |
| Employee or contractor prepares a listed platform's prohibited deployment | Prohibited, including development before deployment. |
| Manual, agent, or subprocessor rewrite using covered source, documentation, or derived design for a prohibited commercial deployment | Prohibited, including C/Go ports and intermediate specifications. |
| Noncommercial rewrite with no commercial development or deployment purpose | Allowed under the noncommercial license. |

## Commercial Use Terms

The commercial license's Appendix A is the fixed platform blacklist. It
contains 89 entries from the [recorded OpenRouter directory snapshot](docs/licensing/openrouter-providers-2026-10-13.json),
retrieved on 2026-10-13 UTC. Four of the directory's 93 entries were excluded:
Azure, Baseten, Google Vertex, and Google AI Studio. Oracle Cloud is also
expressly permitted. The snapshot records provider identifiers and API-only
entries excluded from the list. Future OpenRouter changes do not alter the
license, and no revenue test applies.

The named permitted providers may operate their own deployments on hardware
leased from listed operators if they actually manage serving. This does not
permit deployment inside a listed service, including Bedrock. Routing,
reselling, or relabeling cannot bypass the restriction. Buying NVIDIA
hardware, downloading models, storage, DNS, transit, or unrelated API calls
alone is not platform deployment.

The rewrite condition concerns use of licensed source, documentation, and
design obtained from them. General ideas, algorithms, independent development
without those materials, independently licensed rights, and nonwaivable legal
rights remain unaffected.

## Existing rights and third-party material

The adoption adds the two alternatives only for rights whose holders
authorize them. Michael Feil adopts them for fork rights he owns or is
authorized to license. Another holder's existing INFOIL-only contributions
remain under INFOIL until that holder separately authorizes the alternatives.
Git authorship or maintainer access alone does not establish ownership or
relicensing authority.

Earlier Apache-2.0 and INFOIL grants remain available for the exact material
and rights they cover, including to later recipients. They do not license
new contributions first offered solely under this dual-license adoption.
INFOIL is not a third choice for those new contributions.

The [historical INFOIL adoption notice](LICENSES/INFOIL-1.0-Adoption-Notice.txt),
[INFOIL 1.0 text](LICENSES/INFOIL-1.0.txt), and
[Apache-2.0 text](LICENSES/Apache-2.0.txt) are preserved. The original INFOIL
transition was effective on 2026-10-12, at `infinity_v3-infoil-1.0`, with
pre-adoption base `9486dd9ee89575442fda1a491c381046b37e116e`.

Inherited Hugging Face Text Embeddings Inference, previously Apache-covered
fork contributions, bundled components, and external dependencies retain
their independent grants and notices. Model weights and outputs do not become
covered by either alternative merely through use. See [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and component notices.

## Contributions, redistribution, and packaging

New covered contributions intentionally submitted for inclusion are offered
under both alternatives unless different terms are expressly agreed before
inclusion. Contributors retain ownership. Neither the collective licensor
name nor either license gives anyone blanket authority to issue exceptions
or relicense another holder's work.

Redistributions retain both alternatives, Appendix A, the adoption notice,
and third-party notices, identify changes, and provide covered modifications
under both alternatives. Include corresponding editable source and
build/install instructions, or a free download available for at least three
years after the last corresponding distribution. Private modifications and
network use alone do not require publication. Independent API clients may
retain their own licenses.

The root `LICENSE` bundles both complete alternatives and the unchanged
historical INFOIL/Apache bundle. First-party Cargo packages use this bundle
through `license-file`; independently licensed extension crates retain their
own declarations. Docker builds install the license bundle and component
notices at `/usr/share/licenses/infinity_v3/`; native LibTorch installations
retain their existing `share/licenses/tei_torch/` location.

This is not a complete audit of transitive dependencies, model weights,
CUDA libraries, or base images. Distributors must retain any additional
notices those artifacts require.

References: [SPDX's OR license-choice convention](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/#d42-disjunctive-or-operator),
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), and
[Copyright Office guidance on computer programs](https://www.copyright.gov/register/tx-programs.html).
