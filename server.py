from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import pandas as pd
from io import StringIO

from mcp.server.fastmcp import FastMCP, Context

# OAuth 2.0 scope required for Search Console API
SEARCH_CONSOLE_API_SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Create a simple MCP server
mcp = FastMCP(
    "Search Console Analytics",
    dependencies=["google-api-python-client", "google-auth", "pandas"]
)

@mcp.tool()
def list_verified_sites(context: Context) -> str:
    """List all verified sites connected to your Google Search Console account.
    Returns: A list of site URLs you have access to."""

    try:
        # Get credentials file path from environment variable
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"
        
        # Authenticate using service account credentials
        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
            
        # Set up the Search Console API service
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)
        
        # Get the list of verified sites
        verified_sites_response = search_console_api_service.sites().list().execute()
        verified_site_urls = [site['siteUrl'] for site in verified_sites_response.get('siteEntry', [])]
        
        if not verified_site_urls:
            return "No verified sites found."
        
        return "\n".join([f"- {site_url}" for site_url in verified_site_urls])
        
    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"


@mcp.tool()
def query_search_analytics_data(
    site_url: str, 
    start_date: str, 
    end_date: str, 
    context: Context,
    query_dimensions: list = None,
    search_result_type: str = "web",
    result_row_limit: int = 1000,
    export_as_csv: bool = False   # ✅ NEW!
) -> str:
    """
    Query detailed Search Console performance data for your site.

    Args:
    site_url: Your site URL or domain (e.g., https://www.example.com or sc-domain:example.com)
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)
    query_dimensions: Optional dimensions like query, page, country, device, date
    search_result_type: web, image, video, news, discover, googleNews
    result_row_limit: Number of rows (1–25,000)
    export_as_csv: If True, returns results as CSV

    Returns:
    A formatted table or CSV of clicks, impressions, CTR, and position."""

    try:
        # Get credentials file path from environment variable
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"
        
        # Authenticate using service account credentials
        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
            
        # Set up the Search Console API service
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)
        
        # Validate inputs
        valid_query_dimensions = ['query', 'page', 'country', 'device', 'date']
        if query_dimensions:
            for dimension in query_dimensions:
                if dimension not in valid_query_dimensions:
                    return f"Invalid dimension: {dimension}. Valid dimensions are: {', '.join(valid_query_dimensions)}"
        
        # Validate row_limit
        if result_row_limit < 1 or result_row_limit > 25000:
            return "result_row_limit must be between 1 and 25000"
            
        # Validate search_type
        valid_search_result_types = ['web', 'image', 'video', 'news', 'discover', 'googleNews']
        if search_result_type not in valid_search_result_types:
            return f"Invalid search_result_type: {search_result_type}. Valid types are: {', '.join(valid_search_result_types)}"
        
        # Build the request body
        analytics_request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': query_dimensions or [],
            'searchType': search_result_type,
            'rowLimit': result_row_limit
        }
        
        # Execute the search analytics query
        analytics_api_response = search_console_api_service.searchanalytics().query(siteUrl=site_url, body=analytics_request_body).execute()
        
        if 'rows' not in analytics_api_response or not analytics_api_response['rows']:
            return "No data found for the specified parameters."

        # 🟢 Build DataFrame
        analytics_dataframe = _convert_api_response_to_dataframe(analytics_api_response, query_dimensions)

        if export_as_csv:
            csv_export_data = _export_dataframe_to_csv_format(analytics_dataframe)
            return f"CSV Export:\n\n{csv_export_data}"

        # ✅ Original output stays the same:
        analytics_data_rows = analytics_api_response['rows']
        
        # Format the output based on dimensions
        formatted_output_lines = []
        table_column_headers = []
        
        # Add dimension headers
        if query_dimensions:
            table_column_headers.extend(query_dimensions)
        
        # Add metric headers
        table_column_headers.extend(['clicks', 'impressions', 'ctr', 'position'])
        
        # Add headers to result
        formatted_output_lines.append(" | ".join(table_column_headers))
        formatted_output_lines.append("-" * (sum(len(header) for header in table_column_headers) + 3 * len(table_column_headers)))
        
        # Add data rows
        for data_row in analytics_data_rows:
            formatted_row_data = []
            
            # Add dimension values
            if query_dimensions:
                for dimension_index, dimension_value in enumerate(data_row.get('keys', [])):
                    formatted_row_data.append(dimension_value)
            
            # Add metric values
            formatted_row_data.append(str(data_row.get('clicks', 0)))
            formatted_row_data.append(str(data_row.get('impressions', 0)))
            formatted_row_data.append(f"{data_row.get('ctr', 0) * 100:.2f}%")
            formatted_row_data.append(f"{data_row.get('position', 0):.2f}")
            
            formatted_output_lines.append(" | ".join(formatted_row_data))
        
        return "\n".join(formatted_output_lines)
        
    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

@mcp.tool()
def compare_performance_between_time_periods(
    site_url: str,
    current_period_start_date: str,
    current_period_end_date: str,
    previous_period_start_date: str,
    previous_period_end_date: str,
    context: Context,
    comparison_dimensions: list = None,
    search_result_type: str = "web",
    result_row_limit: int = 1000
) -> str:
    """
    Compare performance metrics between two time periods for your site.

    Args:
    site_url: Your site URL or domain
    current_period_start_date: Start date for current period
    current_period_end_date: End date for current period
    previous_period_start_date: Start date for previous period
    previous_period_end_date: End date for previous period
    comparison_dimensions: Dimensions to compare (query, page, etc.)
    search_result_type: web, image, video, news, discover, googleNews
    result_row_limit: Number of rows

    Returns:
    A table comparing clicks, impressions, CTR, and position between periods.
"""

    try:
        # Get credentials file path from environment variable
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"
        
        # Authenticate using service account credentials
        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
            
        # Set up the Search Console API service
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)
        
        # Validate inputs
        valid_comparison_dimensions = ['query', 'page', 'country', 'device', 'date']
        if comparison_dimensions:
            for dimension in comparison_dimensions:
                if dimension not in valid_comparison_dimensions:
                    return f"Invalid dimension: {dimension}. Valid dimensions are: {', '.join(valid_comparison_dimensions)}"
        
        # Validate row_limit
        if result_row_limit < 1 or result_row_limit > 25000:
            return "result_row_limit must be between 1 and 25000"
            
        # Validate search_type
        valid_search_result_types = ['web', 'image', 'video', 'news', 'discover', 'googleNews']
        if search_result_type not in valid_search_result_types:
            return f"Invalid search_result_type: {search_result_type}. Valid types are: {', '.join(valid_search_result_types)}"
        
        # Build the request body for current period
        current_period_request_body = {
            'startDate': current_period_start_date,
            'endDate': current_period_end_date,
            'dimensions': comparison_dimensions or [],
            'searchType': search_result_type,
            'rowLimit': result_row_limit
        }
        
        # Build the request body for previous period
        previous_period_request_body = {
            'startDate': previous_period_start_date,
            'endDate': previous_period_end_date,
            'dimensions': comparison_dimensions or [],
            'searchType': search_result_type,
            'rowLimit': result_row_limit
        }
        
        # Execute the search analytics query for current period
        current_period_api_response = search_console_api_service.searchanalytics().query(
            siteUrl=site_url, body=current_period_request_body).execute()
        
        # Execute the search analytics query for previous period
        previous_period_api_response = search_console_api_service.searchanalytics().query(
            siteUrl=site_url, body=previous_period_request_body).execute()
        
        # Process and format the results
        if ('rows' not in current_period_api_response or not current_period_api_response['rows']) and \
           ('rows' not in previous_period_api_response or not previous_period_api_response['rows']):
            return "No data found for the specified parameters in either period."
        
        # Convert responses to DataFrames for easier comparison
        current_period_dataframe = _convert_api_response_to_dataframe(current_period_api_response, comparison_dimensions)
        previous_period_dataframe = _convert_api_response_to_dataframe(previous_period_api_response, comparison_dimensions)
        
        # Merge the DataFrames on dimensions
        if comparison_dimensions:
            merged_comparison_dataframe = pd.merge(
                current_period_dataframe, previous_period_dataframe, 
                on=comparison_dimensions, 
                how='outer', 
                suffixes=('_current', '_previous')
            )
        else:
            # If no dimensions, create a single row DataFrame with totals
            merged_comparison_dataframe = pd.DataFrame({
                'clicks_current': [current_period_dataframe['clicks'].sum() if not current_period_dataframe.empty else 0],
                'impressions_current': [current_period_dataframe['impressions'].sum() if not current_period_dataframe.empty else 0],
                'ctr_current': [current_period_dataframe['ctr'].mean() if not current_period_dataframe.empty else 0],
                'position_current': [current_period_dataframe['position'].mean() if not current_period_dataframe.empty else 0],
                'clicks_previous': [previous_period_dataframe['clicks'].sum() if not previous_period_dataframe.empty else 0],
                'impressions_previous': [previous_period_dataframe['impressions'].sum() if not previous_period_dataframe.empty else 0],
                'ctr_previous': [previous_period_dataframe['ctr'].mean() if not previous_period_dataframe.empty else 0],
                'position_previous': [previous_period_dataframe['position'].mean() if not previous_period_dataframe.empty else 0]
            })
        
        # Calculate changes
        merged_comparison_dataframe['clicks_change'] = merged_comparison_dataframe['clicks_current'].fillna(0) - merged_comparison_dataframe['clicks_previous'].fillna(0)
        merged_comparison_dataframe['clicks_change_percentage'] = (
            (merged_comparison_dataframe['clicks_current'].fillna(0) - merged_comparison_dataframe['clicks_previous'].fillna(0)) / 
            merged_comparison_dataframe['clicks_previous'].fillna(1) * 100
        )
        
        merged_comparison_dataframe['impressions_change'] = merged_comparison_dataframe['impressions_current'].fillna(0) - merged_comparison_dataframe['impressions_previous'].fillna(0)
        merged_comparison_dataframe['impressions_change_percentage'] = (
            (merged_comparison_dataframe['impressions_current'].fillna(0) - merged_comparison_dataframe['impressions_previous'].fillna(0)) / 
            merged_comparison_dataframe['impressions_previous'].fillna(1) * 100
        )
        
        merged_comparison_dataframe['ctr_change'] = merged_comparison_dataframe['ctr_current'].fillna(0) - merged_comparison_dataframe['ctr_previous'].fillna(0)
        merged_comparison_dataframe['ctr_change_percentage'] = (
            (merged_comparison_dataframe['ctr_current'].fillna(0) - merged_comparison_dataframe['ctr_previous'].fillna(0)) / 
            merged_comparison_dataframe['ctr_previous'].fillna(0.01) * 100
        )
        
        merged_comparison_dataframe['position_change'] = merged_comparison_dataframe['position_previous'].fillna(0) - merged_comparison_dataframe['position_current'].fillna(0)
        
        # Format the output
        comparison_output_lines = []
        
        # Add period information
        comparison_output_lines.append(f"Comparison between:")
        comparison_output_lines.append(f"Current period: {current_period_start_date} to {current_period_end_date}")
        comparison_output_lines.append(f"Previous period: {previous_period_start_date} to {previous_period_end_date}")
        comparison_output_lines.append("")
        
        # Format the DataFrame as a string table
        if comparison_dimensions:
            # Select columns for display
            display_column_names = comparison_dimensions + [
                'clicks_current', 'clicks_previous', 'clicks_change', 'clicks_change_percentage',
                'impressions_current', 'impressions_previous', 'impressions_change', 'impressions_change_percentage',
                'ctr_current', 'ctr_previous', 'ctr_change', 'ctr_change_percentage',
                'position_current', 'position_previous', 'position_change'
            ]
            display_dataframe = merged_comparison_dataframe[display_column_names].fillna(0)
            
            # Sort by current clicks (descending)
            display_dataframe = display_dataframe.sort_values('clicks_current', ascending=False)
            
            # Format the DataFrame as a string
            string_output_buffer = StringIO()
            
            # Format float columns
            for column_name in display_dataframe.columns:
                if 'ctr' in column_name:
                    display_dataframe[column_name] = display_dataframe[column_name].apply(lambda x: f"{x*100:.2f}%" if 'change_percentage' not in column_name else f"{x:.2f}%")
                elif 'position' in column_name:
                    display_dataframe[column_name] = display_dataframe[column_name].apply(lambda x: f"{x:.2f}")
                elif 'change_percentage' in column_name:
                    display_dataframe[column_name] = display_dataframe[column_name].apply(lambda x: f"{x:.2f}%")
            
            # Convert to string table
            formatted_table_string = display_dataframe.to_string(index=False)
            comparison_output_lines.append(formatted_table_string)
        else:
            # For no dimensions, just show the totals
            summary_data_row = merged_comparison_dataframe.iloc[0]
            comparison_output_lines.append("Overall Metrics:")
            comparison_output_lines.append(f"Clicks: {summary_data_row['clicks_current']:.0f} vs {summary_data_row['clicks_previous']:.0f} ({summary_data_row['clicks_change']:.0f}, {summary_data_row['clicks_change_percentage']:.2f}%)")
            comparison_output_lines.append(f"Impressions: {summary_data_row['impressions_current']:.0f} vs {summary_data_row['impressions_previous']:.0f} ({summary_data_row['impressions_change']:.0f}, {summary_data_row['impressions_change_percentage']:.2f}%)")
            comparison_output_lines.append(f"CTR: {summary_data_row['ctr_current']*100:.2f}% vs {summary_data_row['ctr_previous']*100:.2f}% ({summary_data_row['ctr_change']*100:.2f}%, {summary_data_row['ctr_change_percentage']:.2f}%)")
            comparison_output_lines.append(f"Position: {summary_data_row['position_current']:.2f} vs {summary_data_row['position_previous']:.2f} ({summary_data_row['position_change']:.2f})")
        
        return "\n".join(comparison_output_lines)
        
    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

@mcp.tool()
def get_top_performing_page_content(
    site_url: str,
    start_date: str,
    end_date: str,
    context: Context,
    ranking_metric: str = "clicks",
    results_limit: int = 10
) -> str:
    """
    Get the top performing pages for your site based on a chosen metric.

    Args:
    site_url: Your site URL or domain
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)
    ranking_metric: clicks, impressions, ctr, or position
    results_limit: Number of results to show

    Returns:
    A ranked table of pages with metrics for the selected date range.
    """

    try:
        # Get credentials file path from environment variable
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"
        
        # Authenticate using service account credentials
        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
            
        # Set up the Search Console API service
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)
        
        # Validate metric
        valid_ranking_metrics = ['clicks', 'impressions', 'ctr', 'position']
        if ranking_metric not in valid_ranking_metrics:
            return f"Invalid ranking_metric: {ranking_metric}. Valid metrics are: {', '.join(valid_ranking_metrics)}"
        
        # Build the request body
        page_analytics_request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['page'],
            'searchType': 'web',
            'rowLimit': 1000  # Get a large number to ensure we have enough data
        }
        
        # Execute the search analytics query
        page_analytics_api_response = search_console_api_service.searchanalytics().query(siteUrl=site_url, body=page_analytics_request_body).execute()
        
        # Process and format the results
        if 'rows' not in page_analytics_api_response or not page_analytics_api_response['rows']:
            return "No data found for the specified parameters."
        
        page_analytics_data_rows = page_analytics_api_response['rows']
        
        # Sort by the specified metric
        if ranking_metric == 'position':
            # For position, lower is better
            sorted_page_data_rows = sorted(page_analytics_data_rows, key=lambda x: x.get(ranking_metric, 0))
        else:
            # For other metrics, higher is better
            sorted_page_data_rows = sorted(page_analytics_data_rows, key=lambda x: x.get(ranking_metric, 0), reverse=True)
        
        # Take the top N results
        top_performing_page_rows = sorted_page_data_rows[:results_limit]
        
        # Format the output
        top_pages_output_lines = []
        top_pages_output_lines.append(f"Top {results_limit} Pages by {ranking_metric.capitalize()} ({start_date} to {end_date}):")
        top_pages_output_lines.append("-" * 80)
        
        # Add headers
        top_pages_output_lines.append(f"{'Page':<50} | {'Clicks':<10} | {'Impressions':<12} | {'CTR':<8} | {'Position':<8}")
        top_pages_output_lines.append("-" * 80)
        
        # Add data rows
        for page_data_row in top_performing_page_rows:
            page_url = page_data_row.get('keys', [''])[0]
            page_clicks = page_data_row.get('clicks', 0)
            page_impressions = page_data_row.get('impressions', 0)
            page_ctr = page_data_row.get('ctr', 0) * 100  # Convert to percentage
            page_position = page_data_row.get('position', 0)
            
            top_pages_output_lines.append(f"{page_url[:50]:<50} | {page_clicks:<10.0f} | {page_impressions:<12.0f} | {page_ctr:<8.2f}% | {page_position:<8.2f}")
        
        return "\n".join(top_pages_output_lines)
        
    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

@mcp.tool()
def get_search_performance_trends(
    site_url: str,
    start_date: str,
    end_date: str,
    context: Context,
    trend_interval: str = "week"
) -> str:
    """
    Analyze search performance trends over time for your site.

    Args:
    site_url: Your site URL or domain
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)
    trend_interval: Grouping interval — day, week, or month

    Returns:
    A table showing clicks, impressions, CTR, and position over time."""

    try:
        # Get credentials file path from environment variable
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"
        
        # Authenticate using service account credentials
        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
            
        # Set up the Search Console API service
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)
        
        # Validate interval
        valid_trend_intervals = ['day', 'week', 'month']
        if trend_interval not in valid_trend_intervals:
            return f"Invalid trend_interval: {trend_interval}. Valid intervals are: {', '.join(valid_trend_intervals)}"
        
        # Build the request body
        trends_analytics_request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['date'],
            'searchType': 'web',
            'rowLimit': 1000
        }
        
        # Execute the search analytics query
        trends_analytics_api_response = search_console_api_service.searchanalytics().query(siteUrl=site_url, body=trends_analytics_request_body).execute()
        
        # Process and format the results
        if 'rows' not in trends_analytics_api_response or not trends_analytics_api_response['rows']:
            return "No data found for the specified parameters."
        
        trends_analytics_data_rows = trends_analytics_api_response['rows']
        
        # Convert to DataFrame for easier manipulation
        trends_dataframe = pd.DataFrame([
            {
                'date': data_row['keys'][0],
                'clicks': data_row.get('clicks', 0),
                'impressions': data_row.get('impressions', 0),
                'ctr': data_row.get('ctr', 0),
                'position': data_row.get('position', 0)
            }
            for data_row in trends_analytics_data_rows
        ])
        
        # Convert date string to datetime
        trends_dataframe['date'] = pd.to_datetime(trends_dataframe['date'])
        
        # Group by the specified interval
        if trend_interval == 'day':
            # Already grouped by day
            grouped_trends_dataframe = trends_dataframe
        elif trend_interval == 'week':
            # Group by week
            trends_dataframe['week'] = trends_dataframe['date'].dt.to_period('W').apply(lambda r: r.start_time)
            grouped_trends_dataframe = trends_dataframe.groupby('week').agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'ctr': 'mean',
                'position': 'mean'
            }).reset_index()
            grouped_trends_dataframe.rename(columns={'week': 'date'}, inplace=True)
        elif trend_interval == 'month':
            # Group by month
            trends_dataframe['month'] = trends_dataframe['date'].dt.to_period('M').apply(lambda r: r.start_time)
            grouped_trends_dataframe = trends_dataframe.groupby('month').agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'ctr': 'mean',
                'position': 'mean'
            }).reset_index()
            grouped_trends_dataframe.rename(columns={'month': 'date'}, inplace=True)
        
        # Sort by date
        grouped_trends_dataframe = grouped_trends_dataframe.sort_values('date')
        
        # Format the output
        trends_output_lines = []
        trends_output_lines.append(f"Search Trends by {trend_interval.capitalize()} ({start_date} to {end_date}):")
        trends_output_lines.append("-" * 80)
        
        # Add headers
        trends_output_lines.append(f"{'Date':<12} | {'Clicks':<10} | {'Impressions':<12} | {'CTR':<8} | {'Position':<8}")
        trends_output_lines.append("-" * 80)
        
        # Add data rows
        for _, trend_data_row in grouped_trends_dataframe.iterrows():
            formatted_date_string = trend_data_row['date'].strftime('%Y-%m-%d')
            trend_clicks = trend_data_row['clicks']
            trend_impressions = trend_data_row['impressions']
            trend_ctr = trend_data_row['ctr'] * 100  # Convert to percentage
            trend_position = trend_data_row['position']
            
            trends_output_lines.append(f"{formatted_date_string:<12} | {trend_clicks:<10.0f} | {trend_impressions:<12.0f} | {trend_ctr:<8.2f}% | {trend_position:<8.2f}")
        
        return "\n".join(trends_output_lines)
        
    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

@mcp.tool()
def detect_search_query_cannibalization(
    site_url: str,
    start_date: str,
    end_date: str,
    context: Context,
    result_row_limit: int = 1000
) -> str:
    """
    Detect query cannibalization: find queries ranking with multiple pages.

    Args:
    site_url: Your site URL or domain
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)
    result_row_limit: Number of rows

    Returns:
    A list of queries that appear with more than one page, indicating overlap."""

    try:
        # Load credentials
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS not set"
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"

        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)

        # Query Search Analytics API for query + page combos
        cannibalization_request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query', 'page'],
            'rowLimit': result_row_limit
        }
        cannibalization_api_response = search_console_api_service.searchanalytics().query(siteUrl=site_url, body=cannibalization_request_body).execute()
        if 'rows' not in cannibalization_api_response or not cannibalization_api_response['rows']:
            return "No data found for the specified parameters."

        # Build dataframe
        cannibalization_dataframe = _convert_api_response_to_dataframe(cannibalization_api_response, ['query', 'page'])

        # Find queries with more than 1 unique page
        grouped_queries_by_page_count = cannibalization_dataframe.groupby('query')['page'].nunique().reset_index()
        cannibalized_search_queries = grouped_queries_by_page_count[grouped_queries_by_page_count['page'] > 1]

        if cannibalized_search_queries.empty:
            return "No cannibalization detected — each query maps to only one page."

        # Show details
        cannibalization_output_lines = []
        cannibalization_output_lines.append("Queries ranking with multiple pages:\n")
        for _, cannibalized_query_row in cannibalized_search_queries.iterrows():
            search_query = cannibalized_query_row['query']
            competing_page_urls = cannibalization_dataframe[cannibalization_dataframe['query'] == search_query]['page'].unique()
            formatted_page_list = "\n  - ".join(competing_page_urls)
            cannibalization_output_lines.append(f"🔍 {search_query}\n  - {formatted_page_list}")

        return "\n\n".join(cannibalization_output_lines)

    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

@mcp.tool()
def find_long_tail_keyword_opportunities(
    site_url: str,
    start_date: str,
    end_date: str,
    context: Context,
    minimum_ctr_threshold: float = 5.0,
    maximum_impressions_threshold: int = 100,
    result_row_limit: int = 1000
) -> str:
    """
    Find long-tail keyword opportunities: queries with low impressions but high CTR.

    Args:
    site_url: Your site URL or domain
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)
    minimum_ctr_threshold: Minimum CTR % (e.g., 5.0)
    maximum_impressions_threshold: Maximum impressions to qualify as long-tail
    result_row_limit: Number of rows

    Returns:
    A table of queries matching your long-tail criteria."""

    try:
        google_credentials_file_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not google_credentials_file_path:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS not set"
        if not os.path.exists(google_credentials_file_path):
            return f"Error: Credentials file not found at {google_credentials_file_path}"

        service_account_credentials = service_account.Credentials.from_service_account_file(
            google_credentials_file_path, scopes=SEARCH_CONSOLE_API_SCOPES)
        search_console_api_service = build('webmasters', 'v3', credentials=service_account_credentials)

        long_tail_request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query'],
            'rowLimit': result_row_limit
        }
        long_tail_api_response = search_console_api_service.searchanalytics().query(siteUrl=site_url, body=long_tail_request_body).execute()
        if 'rows' not in long_tail_api_response or not long_tail_api_response['rows']:
            return "No data found for the specified parameters."

        long_tail_data_rows = long_tail_api_response['rows']

        # Filter for low impressions, high CTR
        filtered_long_tail_rows = []
        for data_row in long_tail_data_rows:
            row_impressions = data_row.get('impressions', 0)
            row_ctr_percentage = data_row.get('ctr', 0) * 100  # % format
            if row_impressions <= maximum_impressions_threshold and row_ctr_percentage >= minimum_ctr_threshold:
                filtered_long_tail_rows.append({
                    'query': data_row.get('keys', [''])[0],
                    'clicks': data_row.get('clicks', 0),
                    'impressions': row_impressions,
                    'ctr': f"{row_ctr_percentage:.2f}%",
                    'position': f"{data_row.get('position', 0):.2f}"
                })

        if not filtered_long_tail_rows:
            return "No long-tail keywords found matching the criteria."

        # Format result
        long_tail_output_lines = ["Query | Clicks | Impressions | CTR | Position"]
        long_tail_output_lines.append("-" * 60)
        for long_tail_row in filtered_long_tail_rows:
            long_tail_output_lines.append(f"{long_tail_row['query']} | {long_tail_row['clicks']} | {long_tail_row['impressions']} | {long_tail_row['ctr']} | {long_tail_row['position']}")

        return "\n".join(long_tail_output_lines)

    except Exception as search_console_error:
        return f"Error: {str(search_console_error)}"

def _export_dataframe_to_csv_format(dataframe_to_export: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string."""
    if dataframe_to_export.empty:
        return ""
    return dataframe_to_export.to_csv(index=False)

def _convert_api_response_to_dataframe(api_response_data, response_dimensions):
    """Helper function to convert API response to pandas DataFrame"""
    if 'rows' not in api_response_data or not api_response_data['rows']:
        return pd.DataFrame()
    
    api_response_rows = api_response_data['rows']
    dataframe_data_list = []
    
    for response_row in api_response_rows:
        dataframe_row_data = {}
        
        # Add dimension values
        if response_dimensions:
            for dimension_index, dimension_name in enumerate(response_dimensions):
                if dimension_index < len(response_row.get('keys', [])):
                    dataframe_row_data[dimension_name] = response_row['keys'][dimension_index]
                else:
                    dataframe_row_data[dimension_name] = None
        
        # Add metric values
        dataframe_row_data['clicks'] = response_row.get('clicks', 0)
        dataframe_row_data['impressions'] = response_row.get('impressions', 0)
        dataframe_row_data['ctr'] = response_row.get('ctr', 0)
        dataframe_row_data['position'] = response_row.get('position', 0)
        
        dataframe_data_list.append(dataframe_row_data)
    
    return pd.DataFrame(dataframe_data_list)

if __name__ == "__main__":
    mcp.run()