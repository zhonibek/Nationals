# Microsoft Imagine Cup 2027 plan

Prepared 2026-11-06 on branch `claude/microsoft-ignite-f88450`, cut from `nvidia-nebius-global-ai-hackathon` at `eaa51cb`. The default branch `nvidia-nebius-global-ai-hackathon` stays unchanged while the NVIDIA × Nebius judging runs (December 1–15, 2026).

This file lists what Imagine Cup 2027 requires and checks each requirement against what the project has today. The startup itself is described in [AI_ENGINEERING_LAB_STARTUP_PLAN.md](AI_ENGINEERING_LAB_STARTUP_PLAN.md), and the first product outline is in [`lab/`](../lab/README.md).

**Placeholders.** `[TBD: …]` marks information only the founder can give (names, entity, school, money). Any number without a source is a placeholder. Every other number in this file comes from the Official Rules, the official FAQ or this repository.

## 1. Summary

- **Event.** Imagine Cup 2027 is Microsoft's startup competition for early-stage startups "developing commercially viable software-based solutions using Microsoft Azure" (Official Rules §1). It is not Microsoft Ignite, which is a conference.
- **Deadline.** **February 28, 2027, 23:59 CST.** That is **March 1, 2027, 10:59 in Astana** (UTC+5). Applications have been open since November 2, 2026, and a submitted entry can't be edited.
- **Biggest gaps.** The project has no registered legal entity yet, no Microsoft for Startups account, no Azure subscription and no Azure workloads. Entries need **at least 10 Azure services, each showing at least $1.00 of usage**, on the startup's subscription by the deadline (§8). There is no customer evidence yet beyond an available pilot school.
- **What we already have.** A working, tested software base: the C++20 controller compiled to WebAssembly, the Override rules engine, the Tactical AI with a trained PPO policy, a fail-closed Nemotron tool-calling coordinator, RoboProof evidence records and 619 Node tests. The licenses and notices are also in place.
- **Decision needed first.** Which category to enter: **Deep Tech + Physical AI**, or **Student Startup** (only if every founder is an enrolled student). The two are scored very differently (see §5).

## 2. Sources

| Source | What it gave | Read |
| --- | --- | --- |
| [Imagine Cup 2027 Official Rules](https://aka.ms/ImagineCup2027TCs) (PDF, "Last Updated November 4, 2026", 393,088 bytes, SHA-256 `0a6bd5ae493fbb1a8b3144ec80780da55af91cb842c9af76c321b2bde4eda161`) | Eligibility, entry steps, Azure requirement, scoring tables, prizes, IP and video rules | 2026-11-06 |
| [Official FAQ](https://imaginecup.microsoft.com/en-us/support/faq) | Required materials, the "$1 on 10 services" wording, rolling review | 2026-11-06 |
| [Compete page](https://imaginecup.microsoft.com/en-us/compete) | Category descriptions, timeline, what to prepare | 2026-11-06 |
| [Imagine Cup credits and benefits](https://learn.microsoft.com/en-us/startups/imagine-cup-credits-eligibility) | Credit milestones, business verification steps | 2026-11-06 |
| [Foundry model sponsorship coverage](https://learn.microsoft.com/en-us/startups/benefits/technical-benefits/azure-credits/foundry-model-sponsorship-coverage) | Which Microsoft Foundry models the credits cover | 2026-11-06 |
| [Microsoft Foundry blog, Aug 19, 2026](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/expanding-open-model-choice-in-microsoft-foundry-with-new-deepseek-and-nvidia-ne/4547926) | NVIDIA Nemotron 3.5 Lightning on Foundry (Fireworks on Foundry, tool calling) | 2026-11-06 |

The FAQ says the Official Rules take precedence when the two differ. The rules PDF is Microsoft's document, so it is linked here and not committed. Check again for a newer "Last Updated" date before submitting.

## 3. Key dates

| Stage | Dates (Official Rules §3, §10) | What happens |
| --- | --- | --- |
| 1. Registration, submission, rolling application scoring | Nov 2, 2026 – **Feb 28, 2027, 23:59 CST** (Mar 1, 10:59 Astana) | Award Force account, eligibility check, Azure Subscription ID, answers in English. The **top 30 per category** advance. |
| 2. Expert review | Mar 1–12, 2027 | Pitch deck, demo video, Azure architecture and written answers are scored. The **top 10 per category** advance. |
| 3. Category live interviews | Mar 16–31, 2027 | Recorded interview in English: pitch, product demo and questions. Not scored at this stage. |
| 4. Final judging and selection | April 2027 | Judges score the recordings and materials. Category winners, Top 20, Student Prize and 3 Finalists are chosen. |
| 5. Announcements, finalist preparation | May–June 2027 | Verification, mentoring, rehearsals |
| 6. World Championship | May–June 2027, "in a location in the United States" (§10.6) | Finalists present (not scored). Passports and visas are the entrant's responsibility. |

## 4. Eligibility (Official Rules §4, §14)

| Requirement | Status today | Action |
| --- | --- | --- |
| Every entrant is at least 18 at the time of entry | **Have** (the user confirmed the team is 18+) | Confirm the date of birth of each official member |
| The startup is a **legal entity founded on or after November 2, 2024** | **Missing** | Register a Kazakhstan company, or confirm the details of an existing one: `[TBD: entity]` |
| The startup is **privately held and for-profit**, and **not an educational institution, government organization, consultancy or agency** (Microsoft for Startups rule in §4.1) | **Missing** | Use a commercial company that sells software to schools. It must not be the school itself or a licensed educational organization. |
| Pre-seed to Series B (no Series C or later) and not more than $350,000 in lifetime free Startup Credits | Likely yes, unconfirmed | `[TBD: funding raised so far; earlier Microsoft credits, if any]` |
| Headquartered in a country where Azure services are available; not in a comprehensively sanctioned country or region | Likely yes, unverified. Kazakhstan is not on the rules' list (Cuba, Iran, North Korea, Crimea, Donetsk, Luhansk). | Check that Kazakhstan appears in the Microsoft for Startups sign-up country list, and check the [U.S. Consolidated Screening List](https://www.trade.gov/consolidated-screening-list) |
| Up to **4 official team members**, all named before the deadline. Mentors and advisors can't receive prizes. | **Missing** (no names yet) | `[TBD: team]` |
| One **Authorized Representative** who submits and binds the startup (§4.3) | **Missing** | Usually the founder and director of the entity |
| Not a previous Imagine Cup winner; no Microsoft employees or interns; no family members of competition staff | Unconfirmed | Confirm for each member |
| **Student Startup only:** *all founding team members* are enrolled students at an accredited high school, college or university (home schools count) at some point between Sep 30, 2026 and May 31, 2027 (§4.2). Proof may be requested. | Unknown | `[TBD: student status of each founder]` |

## 5. Category choice

The project fits **Deep Tech + Physical AI**: "startups developing breakthrough technologies that connect advanced software, intelligence, hardware, machines, or physical systems" (Compete page). There is no Education category. Each startup may enter only one category, and Microsoft may reassign an entry (§9).

The categories are scored differently in Stage 1 (§12.2):

| Deep Tech + Physical AI (and the other industry categories): 200 points | Max | Student Startup: 100 points | Max |
| --- | ---: | --- | ---: |
| Startup: Business & Startup Profile (stage, funding stage, **number of paying customers**) | 40 | Business & Startup Profile (stage, intended business model, funding status) | 25 |
| Startup: Product & Technical Approach (cloud platform, AI in the product, which AI models) | 30 | **Product & Technical Approach** (customer validation, technical progress, cloud and AI use, Azure production maturity) | **40** |
| Startup: Problem & Market Opportunity (business model, problem, target customer) | 20 | Team & Execution Potential (makeup of the team, commitment) | 20 |
| Startup: Team & Execution Potential (size, technical composition) | 10 | Problem & Market Opportunity (users or customers, problem, target customer) | 15 |
| Category: Business & Startup Profile (**investor profile, funding stage, capital raised**) | 25 | | |
| Category: Product Deployment & Execution Readiness (scalability, Microsoft and Azure commercial readiness, expected cloud and AI consumption) | 35 | | |
| Category: Market Opportunity & Commercial Potential (**expected ARR, contract value, sales cycle, revenue diversification**) | 40 | | |

**What this means for us.** In Deep Tech, about half of the Stage 1 points reward paying customers, investors, capital raised and revenue. A new, unfunded company with one pilot school will score low there. The Student Startup table puts 40 of 100 points on technical progress and Azure maturity, where this project is strongest.

**Recommendation.** If every founder is an enrolled student, enter **Student Startup**. Otherwise enter **Deep Tech + Physical AI** and push hardest on paying customers and letters of intent before February 28.

The trade-off: the Student Prize is $25,000 cash and $250,000 in Prize Credits, and a Student Startup entry can't become a Finalist (the 1st prize is $250,000 cash and $1,000,000 in credits). Category winners in industry categories get no cash (§13). `[TBD: founder decision]`

## 6. How to enter (§5, §8 and the credits page)

1. The Authorized Representative creates a startup account in Award Force through the [Competition Website](https://www.microsoft.com/en-us/startups/imaginecup2027) and enters company, founder, location, product, consent and source-attribution details. The Microsoft for Startups application also asks for the founder's name, business address, personal LinkedIn, company name, website, country and funding stage.
2. Wait for the eligibility check (business days). Eligible startups get a unique referral code by email.
3. Join Microsoft for Startups with the code. That gives $1,000 in credits, plus $4,000 more and two years in the program when the application is approved.
4. Create the Azure subscription for the solution and note its **Subscription ID**.
5. Complete **business verification** in the Azure portal: legal name, address, country, registration number and tax details exactly as in the official documents. The credits page says this is required for finalist consideration.
6. Choose one category, build the solution, upload the materials, certify and submit. The rules allow one entry per startup and **no edits after submission**.

## 7. Azure requirement and credits

- **Requirement (§8).** By February 28, 2027, the solution must use **at least 10 Azure services** on the submitted subscription, and Azure must show **at least $1.00 of usage on each one**. The FAQ adds that each workload must stay at or above $1 "starting in February 28, 2027 and as long as you are still actively competing."
- **Scoring.** The Stage 4 factor "Azure architecture and production workloads" is worth 6 of 100 points. Azure also feeds Stage 1 ("cloud platform in use", "Azure production maturity") and the Stage 2 architecture items (40 of 100 points).
- **Credit milestones (§6).**

| Credits | Condition |
| --- | --- |
| $1,000 | Referral code redeemed |
| $5,000 | Microsoft for Startups application approved (2 years in the program) |
| $25,000 | 5 or more eligible workloads for 60 continuous days |
| $50,000 | 7 or more eligible workloads for 60 continuous days |
| $150,000 | 10 or more eligible workloads and $3,000 per month of Azure usage for 60 continuous days |

- **What credits don't cover.** Credits cover Microsoft Foundry models "sold and billed directly by Azure". Models billed by partners or through the Microsoft Marketplace are pay-as-you-go. NVIDIA Nemotron 3.5 Lightning is offered through **Fireworks on Foundry** (partner-hosted), so it most likely isn't covered. Confirm in the Foundry portal's "Direct from Azure" collection.
- **Card needed.** A valid credit card on file is used for identity verification and for charges the credits don't cover.
- **Free allowances.** Some services have free monthly allowances (for example log ingestion, serverless grants and identity sign-ins). A service under its free allowance shows $0 and won't count. The startup plan (§8) marks which services need care, and the usage must be checked in **Azure Cost Management by February 14, 2027**.

## 8. Submission materials

The rules (§8) require Stage 1 answers in English by February 28. The pitch deck, video, architecture overview and written responses are due "if the Startup advances to Stage 2 … by the deadline Microsoft communicates." The FAQ lists all of them as due by February 28. **Plan to have everything ready by February 28.**

| Material | Stage 2 points | Have | Missing | Drafted in |
| --- | ---: | --- | --- | --- |
| Answers to all submission questions (English) | Stage 1 | English docs to draw from (README, HACKATHON_POST.md) | The answers themselves | `[TBD]` after Award Force opens the form |
| Pitch deck: problem, solution, market, traction, business model, competition, team, metrics | 20 | Hackathon write-up and architecture facts | A startup deck for AI Engineering Lab | Startup plan §1–§7, §11–§12 |
| Product demo video, **2–3 minutes**, the product in operation | 15 | A hackathon demo film exists on another branch (2:55) | A new video of the *learning product*, filmed and edited by the team (§15: the video "must be solely the work of the team") | Startup plan §10 (shot list to do) |
| Azure architecture diagram or technical overview (components, data flows, AI, security boundaries, dependencies) | 20 | Today's architecture (README) | The Azure version | Startup plan §8 (draft diagram) |
| Production architecture response (scalability, reliability, security, observability, production readiness) | 20 | Fail-closed AI, tests, provenance | Production deployment, monitoring, incident record | Startup plan §8–§9 |
| Customer differentiation response (why us; what is hard to copy) | 10 | Unique stack (same C++ on robot and browser, rules engine, measured AI) | Customer evidence | Startup plan §3 |
| Microsoft growth-alignment response | 15 | — | The answer | Startup plan §13 |

## 9. Final judging (Stage 4, §12.5) and where we stand

Each of the four criteria is worth 25 points (100 total). Judges look for evidence, not plans.

| Criterion | Factor (max points) | Our evidence today | Gap to close by March |
| --- | --- | --- | --- |
| Enterprise customer validation | Adoption feasibility (6), measurable outcomes (7), customer references (6), retention (6) | Pilot school is available, but not named or signed | A signed pilot or LOI, a baseline and a post-pilot measurement, a reference contact with permission. Retention: say candidly that the pilot is under 12 months old. |
| Investability | Market opportunity (6), traction and revenue growth (7), funding readiness (6), scalability and defensibility (6) | — | A bottom-up 3-year market estimate, one traction metric observed monthly (6 observations are expected; we can have about 3–4), funding plan, unit economics |
| Technical excellence | Azure architecture and production workloads (6), reliability and operational maturity (6), security (7), scalability and deployment (6) | Strong software base, fail-closed AI, 619 tests | Live Azure workload, a documented incident or failure drill with measured recovery, implemented child-safety and AI controls, a 10× load estimate with cost per active user |
| Customer value | Problem and urgency (7), industry fit (6), go-to-market fit (6), enterprise sales potential (6) | — | Named current alternative, cost of doing nothing, buyer and budget owner at the school, pilot-to-paid conversion |

Ties in Top 20 selection are broken by technical excellence, then customer value, then enterprise customer validation, then investability (§10.4).

## 10. Originality, rights and language (§15, §17, §18)

- **Original work.** The entry must be the team's own work, and third-party material needs licenses *and* cited permissions. `LICENSE`, `THIRD_PARTY_NOTICES.md` and the Pedro Pathing `NOTICE.md` files already list LemLib (MIT), Pedro Pathing (BSD-3-Clause; our C++ rewrite of the Java library), PROS and Eigen (MPL-2.0), {fmt} (MIT), Three.js (MIT) and the NVIDIA model terms. Add any new Azure SDKs, lesson images, music and fonts to the same file.
- **Copyright holder name.** `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md`, `HACKATHON_POST.md` and both Pedro `NOTICE.md` files name "Kassymkhan", which is not the founder's name. Fix this on this branch once the real name is known. The default branch should wait until hackathon judging ends.
- **IP ownership.** The startup and its members must own or license everything (§4.3, §14), so the founder should **assign the project IP to the new company**. Microsoft doesn't take ownership, but it gets a broad license to use and publicly post the entry (§18).
- **No Microsoft property.** Don't use Microsoft internal resources or the work of Microsoft employees (§15).
- **Video.** The demo video must be filmed and edited by the team (§15). Don't reuse the hackathon film.
- **English.** All materials, interviews and presentations must be in English (§15). The product itself can be in Kazakh, Russian and English.
- **Earlier hackathon entry.** The Imagine Cup rules don't forbid reusing work from another competition. The NVIDIA × Nebius entry is MIT-licensed by the team. Still, re-read the [hackathon rules](https://nebiusglobalaihackathon.devpost.com/rules) for any license granted to the sponsors before submitting. `[TBD: check]`

## 11. Master checklist

Status: **Have**, **Partly** or **Missing**. Owner `F` is the founder. Dates are targets, not rules, except the deadline.

| # | Requirement | Status | What exists | Next action | Target date |
| --- | --- | --- | --- | --- | --- |
| E1 | All entrants 18+ | Have | Confirmed by the user | Collect dates of birth | Nov 15 |
| E2 | Legal entity founded on or after Nov 2, 2024, for-profit, not an educational institution | Missing | — | Register or confirm the Kazakhstan company (F) | Nov 22 |
| E3 | Up to 4 official members and an Authorized Representative | Missing | — | Name them (F) | Nov 15 |
| E4 | Category decided | Missing | Analysis in §5 | Decide (F) | Nov 15 |
| E5 | Kazakhstan eligible for Microsoft for Startups and Azure | Partly | Not on the rules' sanctions list | Check the sign-up country list | Nov 15 |
| R1 | Award Force account and eligibility check | Missing | — | Create the account (F) | Nov 22 |
| R2 | Microsoft for Startups enrollment (referral code) | Missing | — | Redeem the code (F) | Nov 29 |
| R3 | Azure Subscription ID | Missing | — | Create the subscription (F) | Nov 29 |
| R4 | Business verification in the Azure portal | Missing | — | Enter the registry data exactly (F) | Dec 13 |
| A1 | 10+ Azure services, each at $1.00 or more, used by the solution | Missing | Service plan in the startup plan §8 | Deploy, then check in Cost Management | Feb 14 |
| A2 | Live production workload on Azure | Missing | — | Pilot runs on the Azure deployment | Jan 11 |
| A3 | Microsoft Foundry model path | Missing | Nebius-only client (`roboproof/nemotron.js` accepts only Nebius or loopback hosts) | Add a Foundry provider and compare with the existing harness | Dec 20 |
| P1 | Kids' learning app (MVP) | Missing | Simulator, rules engine, Tactical AI, Nemotron coordinator, RoboProof | Build lessons 1–3 (see `lab/`) | Dec 20 |
| P2 | Security and child-safety controls in place | Partly | Fail-closed tool calls, allowlists, server-side keys | Identity, Content Safety, pseudonymous data, privacy notice, consent | Jan 11 |
| P3 | Reliability evidence | Missing | Tests, provenance | Monitoring, alerts and one failure drill with measured recovery | Feb 7 |
| C1 | Pilot agreement or LOI with the school | Missing | School available | Sign it (F) | Dec 13 |
| C2 | Baseline and post-pilot measurement | Missing | Metric plan in the startup plan §5 | Measure in January and February | Feb 21 |
| C3 | Customer reference with permission to contact | Missing | — | Ask the pilot school (F) | Feb 21 |
| C4 | Monthly traction metric | Missing | — | Record on Dec 31, Jan 31 and Feb 28 | Monthly |
| C5 | Pricing, market size and funding plan | Missing | Method in the startup plan §6–§7, §12 | Fill the `[TBD]` values (F) | Feb 7 |
| M1 | Answers in English | Missing | — | Draft them in Award Force | Feb 21 |
| M2 | Pitch deck | Missing | Startup plan | Build the deck | Feb 21 |
| M3 | 2–3 minute demo video, team-made | Missing | — | Film and edit | Feb 21 |
| M4 | Azure architecture diagram | Partly | Draft in the startup plan §8 | Update it from the real deployment | Feb 14 |
| M5 | Production architecture, differentiation and Microsoft growth answers | Partly | Drafts in the startup plan | Finalize | Feb 21 |
| L1 | Third-party licenses cited | Have | `THIRD_PARTY_NOTICES.md` | Add new dependencies as they come in | Ongoing |
| L2 | Correct copyright holder name; IP assigned to the company | Missing | — | Fix the name in 5 files; sign the assignment (F) | Dec 20 |
| S1 | **Submit** | Missing | — | Submit early. No edits after submission. | **Feb 25** (rule: Feb 28, 23:59 CST) |
| S2 | English interview ready | Missing | — | Rehearse the pitch, demo and Q&A | Mar 15 |
| S3 | Passports and visas for a U.S. final | Unknown | — | Check passports (F) | Apr 30 |

## 12. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Entity registration or business verification is slow | No Azure subscription, so no entry | Start this week. Copy registry data exactly, including leading zeros. |
| A service stays under its free allowance and shows $0 | The 10-service rule fails | Choose services whose real usage exceeds $1. Check Cost Management on Feb 14. |
| Azure bill grows (partner-billed Nemotron, GPU training) | Cash spend | Budget alerts. Training only with an approved budget, bounded jobs and scale-to-zero (see `AGENTS.md`). |
| Children's personal data handled wrongly | Legal risk, loss of trust, lost Security points | Pseudonymous student IDs, no names in the cloud until legal review, parental consent. Kazakhstan's personal-data law has localization rules: `[TBD: confirm with a Kazakhstan lawyer]`. |
| Too little customer evidence by March | Low Stage 4 scores | Start the pilot by January 11 and measure from day one. Be candid about short history. |
| Pitch claims more than the evidence | Disqualification risk (§10.2: material misrepresentation) and lost credibility | Follow the RoboProof rule: only measured results are claimed. For example, the PPO policy does *not* beat the loading-zone tactic (201.5 vs 244.5). |
| NVIDIA hackathon branch changed during judging | Hurts the other entry | Work only on this branch until December 15 |

## 13. Questions for the founder

1. **Your full legal name** (as in your ID and the company documents) and your LinkedIn URL. The Git user name "Kassymkhan" will not be used.
2. **Team.** Up to 4 official members: names, roles, confirmed 18+, and whether *every* founder is an enrolled student through May 31, 2027 (this decides whether Student Startup is possible).
3. **Category.** Student Startup or Deep Tech + Physical AI (see §5)?
4. **Kazakhstan entity.** Does it exist yet? Legal form, registered name, BIN, registration date, registered address, director (Authorized Representative), owners, any funding raised, any earlier Microsoft credits, website or domain. It must be a for-profit company, not an educational organization.
5. **Pilot school.** Name, city, public or private, contact person and role, who approves budgets, number of students and age range, VEX V5 kits on hand, earliest start date. Will they sign a pilot agreement or LOI, pay after the pilot, and act as a named reference? How is parental consent handled?
6. **Budget.** Monthly Azure spend you can cover beyond credits, a card for Azure verification, a GPU training budget (needs your approval under `AGENTS.md`), and travel money for a possible U.S. final.
7. **Product.** Which languages the school needs (Kazakh, Russian, English), and what price the school says it would pay.

## 14. Package contents

- This plan: requirements, scoring and checklist.
- [AI_ENGINEERING_LAB_STARTUP_PLAN.md](AI_ENGINEERING_LAB_STARTUP_PLAN.md): product, users, pilot, business model, Azure architecture, roadmap and team.
- [`lab/`](../lab/README.md): MVP folder structure for the kids' learning app, built on the existing simulator. Proposal only; no app code yet.
