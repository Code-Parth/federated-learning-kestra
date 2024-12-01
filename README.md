# Federated Credit Scoring with Kestra

### Prerequisites
- Docker
- Docker Compose

### Installation
1. Clone the repository
2. Run `docker-compose up --build`

### Access
- Kestra Dashboard: http://localhost:8080
- Workflow: `flows/federated-learning-workflow.yml`

## Workflow Details
- Builds Docker image for Federated Learning
- Runs Federated Learning simulation
- Stores results in `results/` directory

## Configurable Parameters
- `NUM_INSTITUTIONS`: Number of simulated institutions
- `NUM_SAMPLES`: Samples per institution
- `RANDOM_SEED`: Reproducibility seed
