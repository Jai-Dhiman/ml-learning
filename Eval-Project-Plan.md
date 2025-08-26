Prompt:

I need you to create a complete embedding model comparison harness project. This is a portfolio project to showcase "high-throughput experimentation + evaluation" for
  AI/ML engineering roles at top tech companies.

  **BACKGROUND CONTEXT:**
  - Current experience: Built semantic search with embeddings + Qdrant at Capture, including custom evaluation harnesses
  - Tech stack expertise: Python, PyTorch, HuggingFace, FastAPI, MLflow, React/TypeScript, currently using Voyage embeddings
  - Target: AI/ML Engineer roles (OpenAI, Anthropic, Google, Meta)
  - Infrastructure: Local MacBook → Docker → GCP + Kubernetes

  **PROJECT REQUIREMENTS:**
  Create a modular system that can:
  1. Load any HuggingFace embedding model + API models (Voyage, OpenAI, Cohere)
  2. Run systematic evaluations on multiple benchmarks
  3. Generate comprehensive dashboards and reports
  4. Scale from local development to production deployment

  **EVALUATION FOCUS (prioritized for ML engineering impact):**
  1. Information Retrieval (BEIR: MS-MARCO, Natural Questions, SciFact)
  2. Few-Shot Classification (Banking77, Intent Detection, News)
  3. Domain Robustness & Distribution Shift analysis
  4. Efficiency Analysis (latency, memory, throughput)
  5. Future: Text + Multimodal capability

  **DELIVERABLES TO CREATE:**

  **1. PROJECT SETUP:**
  - Create project structure with proper directories
  - Write comprehensive CLAUDE.md file following these specifications:
  CLAUDE.md Requirements:

  - Project overview and goals
  - Complete tech stack documentation
  - Directory structure explanation
  - Development commands (make setup, run-backend, run-frontend, test, lint, type-check)
  - Docker commands (docker-build, docker-up, docker-test)
  - Deployment commands (deploy-staging, deploy-prod)
  - Code quality standards (type hints, 85% coverage, documentation, linting)
  - Implementation phases and priorities
  - Assistance guidelines for future Claude sessions

  **2. PRD (Product Requirements Document):**
  Create a detailed PRD that includes:
  - Executive summary and success metrics
  - Functional requirements (core features)
  - Technical requirements (performance, scalability)
  - Phase-by-phase implementation plan with scope boundaries
  - Non-functional requirements (security, monitoring, observability)
  - Success criteria and portfolio impact metrics
  - Explicit scope limitations to prevent feature creep

  **3. TECHNICAL ARCHITECTURE:**
  - System design with component interactions
  - API specifications and data schemas
  - Database design for experiments and results
  - Model integration interfaces
  - Deployment architecture (local → cloud)

  **4. IMPLEMENTATION FOUNDATION:**
  - Core project structure with placeholder files
  - Configuration management system
  - Basic model loading interface
  - Simple evaluation framework skeleton
  - Docker setup for development
  - Basic FastAPI backend structure
  - React frontend scaffold
  - Testing framework setup

  **CRITICAL SUCCESS FACTORS:**
  - This must demonstrate advanced ML engineering capabilities beyond basic model comparison
  - Show systematic approach to experimentation and evaluation
  - Exhibit production-ready architecture and code quality
  - Provide concrete, impressive metrics for portfolio presentation
  - Create extensible foundation for future ML projects

  **TECHNICAL SPECIFICATIONS:**
  - Backend: FastAPI with async processing, task queues, result caching
  - Frontend: React dashboard for experiment management and visualization
  - ML Pipeline: HuggingFace integration, custom evaluation metrics, result storage
  - Infrastructure: Local development, Docker containers, GCP deployment ready
  - Monitoring: MLflow/W&B integration, performance tracking, error handling

  Please create all necessary files, documentation, and project structure to give me a solid foundation for implementing this embedding evaluation harness. Focus on
  creating a professional, well-documented project that showcases ML engineering best practices.