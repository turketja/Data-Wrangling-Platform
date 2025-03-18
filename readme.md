# Data Analysis Platform with Integrated Chatbot

A unified environment for data analysis, data wrangling, and data cleaning with an integrated chatbot to quickly query and understand your data.

## Features

- **Unified Web Interface**: Single entry point for all data analysis needs
- **Data Storage**: PostgreSQL for structured data and MinIO (S3-compatible) for files
- **Interactive Analysis**: JupyterLab for advanced data exploration
- **Chatbot Integration**: Ask questions about your data in natural language
- **Data Visualization**: Built-in visualization capabilities
- **Docker-based**: Easy setup and deployment

## Components

- **UI**: Streamlit-based unified interface
- **JupyterLab**: For interactive notebooks and advanced analysis
- **PostgreSQL**: Relational database for structured data
- **MinIO**: S3-compatible object storage for files
- **Chatbot**: FastAPI service for natural language data queries

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/data-analysis-platform.git
   cd data-analysis-platform
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Access the platform:
   - Main UI: http://localhost:8501
   - JupyterLab: http://localhost:8888
   - MinIO Console: http://localhost:9001 (login with minioadmin/minioadmin)

## Usage

### Data Upload

1. Navigate to the "Data Upload" page
2. Upload your data files (CSV, Excel, JSON, etc.)
3. Optionally save structured data to PostgreSQL

### Data Exploration

- Use the "File Explorer" to browse and preview uploaded files
- Use the "Data Analysis" page for quick visualizations and statistics

### Chatbot Interaction

1. Go to the "Chat" page
2. Select a file or database table as context
3. Ask questions about your data in natural language

### Advanced Analysis

- Open JupyterLab for advanced analysis
- Your data files are available in the `/home/jovyan/data` directory
- Connect to PostgreSQL using `postgresql://datauser:datapass@postgres:5432/datadb`

## Example Queries for the Chatbot

- "What columns are in this dataset?"
- "How many rows does this data have?"
- "Show me the summary statistics"
- "What is the maximum value of [column_name]?"
- "Are there any missing values?"
- "What is the average of [column_name]?"

## Customization

### Adding More Analysis Tools

Edit the `docker-compose.yml` file to add more services. For example, to add RStudio:

```yaml
rstudio:
  image: rocker/rstudio:latest
  ports:
    - "8787:8787"
  volumes:
    - ./data:/home/rstudio/data
  environment:
    - PASSWORD=yourpassword
```

### Enhancing the Chatbot

The chatbot service in `chatbot/app.py` can be extended with more advanced natural language processing capabilities or by integrating with external AI services.

## Troubleshooting

- **Services not starting**: Check Docker logs with `docker-compose logs`
- **Cannot upload files**: Ensure MinIO is running with `docker-compose ps`
- **Database connection issues**: Verify PostgreSQL is running and credentials are correct

## Maintenance

- **Backup data**: 
  ```
  docker-compose exec postgres pg_dump -U datauser datadb > backup.sql
  ```
- **Update services**:
  ```
  docker-compose pull
  docker-compose up -d
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
