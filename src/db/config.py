import os

# Local dev default matches the container started for this project (see README, Database
# section). Overridable via env var for anything else (CI, Docker Compose service name, prod).
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://credit_risk:credit_risk@localhost:5432/credit_risk")
