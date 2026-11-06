# lab/azure: infrastructure plan

Planned. Nothing is deployed yet, and no Azure subscription exists yet.

- **Infrastructure as code:** Bicep files for the services listed in the [startup plan §8](../../docs/AI_ENGINEERING_LAB_STARTUP_PLAN.md#8-azure-architecture), one environment per stage (`dev`, `pilot`).
- **Identity:** managed identities for every service-to-service call; no keys in code or in Git.
- **Imagine Cup rule:** at least 10 Azure services, each showing at least $1.00 of usage, by February 28, 2027 (Official Rules §8).
  - Track every service in Azure Cost Management.
  - Plan for 12 or more, because free allowances show $0.
  - Ask `imagine@microsoft.com` how unique services are counted.
- **Budget:** `[TBD: monthly limit]`, with alerts at `[TBD]`. GPU jobs only with an approved budget, bounded runtime and scale-to-zero.
- **Region:** `[TBD: after the legal review]`.
