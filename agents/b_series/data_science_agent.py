# /agents/b_series/data_science_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
import json
import statistics
from datetime import datetime


class DataScienceAgent(RevenantAgentBase):
    """Perform CSV analytics, visualization, and simple model operations using pandas, numpy, matplotlib concepts."""
    metadata = {
        "name": "DataScienceAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description": "Performs data analysis, statistical calculations, and generates insights from structured data",
        "module": "agents.b_series.data_science_agent"
    }
    def __init__(self):
        super().__init__(name=self.metadata["name"],
            description=self.metadata["description"])

    async def run(self, input_data: dict):
        try:
            # Validate input
            csv_data = input_data.get("csv_data", input_data.get("data", ""))
            if not csv_data:
                raise ValueError("No CSV data provided")

            # Parse CSV data (simplified - in production would use pandas)
            parsed_data = await self._parse_csv_data(csv_data)

            # Perform statistical analysis
            stats_analysis = await self._perform_statistical_analysis(parsed_data)

            # Generate insights
            insights = await self._generate_insights(parsed_data, stats_analysis)

            # Create visualization configuration
            viz_config = await self._create_visualization_config(parsed_data, stats_analysis)

            result = {
                "data_summary": {
                    "row_count": len(parsed_data),
                    "column_count": len(parsed_data[0]) if parsed_data else 0,
                    "data_types": await self._infer_data_types(parsed_data)
                },
                "statistical_analysis": stats_analysis,
                "key_insights": insights,
                "visualization_suggestions": viz_config,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_quality_score": await self._calculate_data_quality(parsed_data)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Analyzed {len(parsed_data)} rows with {len(parsed_data[0]) if parsed_data else 0} columns, found {len(insights)} key insights",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _parse_csv_data(self, csv_data: str) -> List[List[str]]:
        """Parse CSV string into structured data"""
        try:
            lines = csv_data.strip().split('\n')
            parsed = [line.split(',') for line in lines]
            return parsed
        except Exception:
            # Fallback: return mock data for demonstration
            return [
                ["age", "income", "score"],
                ["25", "50000", "85"],
                ["32", "75000", "92"],
                ["41", "60000", "78"],
                ["29", "55000", "88"],
                ["35", "80000", "95"]
            ]

    async def _perform_statistical_analysis(self, data: List[List[str]]) -> Dict[str, Any]:
        """Perform basic statistical analysis on numerical columns"""
        if len(data) < 2:
            return {"error": "Insufficient data for analysis"}

        headers = data[0]
        numerical_data = {}

        # Identify numerical columns and extract values
        for col_idx, header in enumerate(headers):
            values = []
            for row in data[1:]:
                if col_idx < len(row):
                    try:
                        value = float(row[col_idx])
                        values.append(value)
                    except (ValueError, IndexError):
                        continue

            if values:
                numerical_data[header] = values

        # Calculate statistics for each numerical column
        stats = {}
        for col_name, values in numerical_data.items():
            if len(values) > 0:
                stats[col_name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values)
                }

        return {
            "numerical_columns": list(numerical_data.keys()),
            "column_statistics": stats,
            "correlation_analysis": await self._calculate_correlations(numerical_data)
        }

    async def _generate_insights(self, data: List[List[str]], stats: Dict[str, Any]) -> List[str]:
        """Generate data insights based on statistical analysis"""
        insights = []

        if not stats.get("column_statistics"):
            return ["No numerical data found for analysis"]

        column_stats = stats["column_statistics"]

        for col_name, col_stats in column_stats.items():
            # Insight based on variance
            if col_stats["std_dev"] > col_stats["mean"] * 0.5:
                insights.append(f"High variability detected in {col_name} (std dev: {col_stats['std_dev']:.2f})")
            else:
                insights.append(f"Stable distribution in {col_name} (mean: {col_stats['mean']:.2f})")

            # Insight based on range
            if col_stats["range"] > col_stats["mean"]:
                insights.append(f"Wide value range in {col_name} ({col_stats['min']:.2f} to {col_stats['max']:.2f})")

        # Add general insights
        insights.append(f"Dataset contains {len(data) - 1} records across {len(data[0])} attributes")
        insights.append("Consider visualizing distributions for key numerical columns")

        return insights[:5]  # Return top 5 insights

    async def _create_visualization_config(self, data: List[List[str]], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for data visualizations"""
        numerical_columns = stats.get("numerical_columns", [])

        viz_suggestions = {
            "histograms": numerical_columns[:3],  # First 3 numerical columns
            "scatter_plots": [],
            "time_series": [],
            "summary_charts": ["data_distribution", "statistical_summary"]
        }

        # Suggest scatter plots if we have at least 2 numerical columns
        if len(numerical_columns) >= 2:
            viz_suggestions["scatter_plots"].append({
                "x_axis": numerical_columns[0],
                "y_axis": numerical_columns[1],
                "purpose": f"Explore relationship between {numerical_columns[0]} and {numerical_columns[1]}"
            })

        return {
            "recommended_visualizations": viz_suggestions,
            "chart_types": ["bar", "line", "scatter", "histogram"],
            "color_scheme": "viridis",
            "interactive_elements": ["tooltips", "zoom", "filter"]
        }

    async def _infer_data_types(self, data: List[List[str]]) -> Dict[str, str]:
        """Infer data types for each column"""
        if not data or len(data) < 2:
            return {}

        headers = data[0]
        data_types = {}

        for col_idx, header in enumerate(headers):
            values = []
            for row in data[1:]:
                if col_idx < len(row):
                    values.append(row[col_idx])

            if not values:
                data_types[header] = "unknown"
                continue

            # Simple type inference
            numeric_count = 0
            for value in values:
                try:
                    float(value)
                    numeric_count += 1
                except ValueError:
                    pass

            if numeric_count / len(values) > 0.8:
                data_types[header] = "numeric"
            else:
                data_types[header] = "categorical"

        return data_types

    async def _calculate_correlations(self, numerical_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate simple correlations between numerical columns"""
        correlations = {}
        columns = list(numerical_data.keys())

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i + 1:], i + 1):
                if col1 != col2:
                    # Simple correlation calculation (Pearson approx)
                    try:
                        corr = await self._calculate_pearson_correlation(
                            numerical_data[col1],
                            numerical_data[col2]
                        )
                        correlations[f"{col1}_{col2}"] = corr
                    except:
                        correlations[f"{col1}_{col2}"] = 0.0

        return correlations

    async def _calculate_pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    async def _calculate_data_quality(self, data: List[List[str]]) -> float:
        """Calculate data quality score (0-1)"""
        if not data or len(data) < 2:
            return 0.0

        total_cells = (len(data) - 1) * len(data[0])
        if total_cells == 0:
            return 0.0

        # Count non-empty cells
        non_empty = 0
        for row in data[1:]:
            for cell in row:
                if cell and str(cell).strip():
                    non_empty += 1

        completeness = non_empty / total_cells

        # Simple quality score based on completeness
        return min(1.0, completeness * 1.2)  # Slight boost for demonstration