# lab/classroom: classes and progress

Planned. This folder holds the data model behind the teacher view.

- **Entities:** school, class, teacher, pseudonymous student, lesson attempt, check result, and bench run approval.
- **Students:** each has a random ID and a class code. Real names stay on the school's roster, not in the cloud, until a Kazakhstan legal review says otherwise `[TBD]`.
- **Teachers:** sign in through Microsoft Entra External ID.
- **Storage:** Azure Database for PostgreSQL in production, and a local file store for development.
- **Export:** a class report for the school, with measured outcomes only.
- **Retention and deletion:** `[TBD: agree with the pilot school]`.
