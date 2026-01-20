# Multi-Context Processing (MCP) System

A powerful system for multi-context data processing across PostgreSQL databases and GitHub repositories, powered by AI analysis through OpenRouter and DeepSeek Chat v3 model.

## System Architecture

The MCP system consists of three main components:

1. **PostgreSQL Server** - Handles database queries and natural language to SQL conversion
2. **GitHub Server** - Provides GitHub repository analysis and exploration
3. **Streamlit Web Interface** - Unified dashboard for interacting with both services

### Key Features

- **Natural Language to SQL** - Convert English descriptions into executable SQL queries
- **GitHub Repository Analysis** - Analyze repositories, find similar projects, and explore codebase structure
- **AI-Powered Data Analysis** - Analyze query results with LLM models
- **Interactive Dashboard** - User-friendly Streamlit interface for all operations

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- GitHub account and API token (for GitHub features)
- OpenRouter API key

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=your_database
DB_PORT=5432

# GitHub Configuration
GITHUB_TOKEN=your_github_token

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the setup script to initialize the database (if needed):

```bash
python setup_database.py
```

## Usage

### Starting the System

Run the Streamlit application:

```bash
streamlit run streamlit_mcp_app.py
```

The application will automatically start the required servers or you can start them manually:

- PostgreSQL Server: `python mysql_mcp_server.py --port 8004`
- GitHub Server: `python github_mcp_server.py --port 8002`

### Using the Dashboard

1. **PostgreSQL Explorer**
   - List database tables
   - Execute SQL queries directly
   - Use natural language to generate SQL
   - View table schemas

2. **GitHub Explorer**
   - Browse your repositories
   - Analyze repository structure and code
   - Find similar repositories
   - Search for specific repositories

3. **Data Analysis**
   - Analyze query results with AI
   - Ask follow-up questions
   - Save and load previous analysis results

## Troubleshooting

### Port Conflicts

If you encounter port binding errors, specify different ports when starting the servers:

```bash
python github_mcp_server.py --port 8010
python mysql_mcp_server.py --port 8011
```

Then update the connection URLs in the Streamlit app through environment variables:

```bash
GITHUB_SERVER_HOST=localhost GITHUB_SERVER_PORT=8010 POSTGRES_SERVER_HOST=localhost POSTGRES_SERVER_PORT=8011 streamlit run streamlit_mcp_app.py
```

### API Connection Issues

- Ensure your `.env` file contains the proper API keys
- Check if the OpenRouter API is accessible from your network
- Verify that your GitHub token has the necessary permissions

## Security Considerations

- This system is designed for local development and internal use only
- Add authentication if deploying to a shared environment
- Consider encrypting sensitive API keys

## License

This project is licensed under the MIT License - see the LICENSE file for details. 