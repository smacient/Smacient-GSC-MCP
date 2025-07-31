# MCP Server For Google Search Console

This is a Model Context Protocol (MCP) server with tools to retrieve information from Google Search Console. It integrates easily with claude desktop and gives helpful business intelligence feedback based on your data

## Setup Steps

1.  ## Initialize the project 
    #### launch CMD where you want to keep your project, and clone the repo, enter project folder
```bash
    git clone https://github.com/smacient/Smacient-GSC-MCP
    cd Smacient-GSC-MCP
```

2.  ## Create virtual environment and activate it
```bash
    uv venv
    .venv\Scripts\activate
  ```

3.  ## Install dependencies:
```bash
    uv sync
```

4.  ## Setup environment
- set up your google account credentials and download the json file. rename it credentials.json
- create a `.env` file inside the `localserver` folder
- add the correct path to your google credentials in the env file

    `GOOGLE_APPLICATION_CREDENTIALS="C:\\Users\\path to\\credentials.json"`
