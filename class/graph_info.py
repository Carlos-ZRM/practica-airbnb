import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_boxplot, geom_density, theme_bw, labs, geom_histogram
import io
import base64
from datetime import datetime


class graph_info:
    """
    Automatically analyze and visualize DataFrames with automatic variable type detection.
    Generates comprehensive HTML report with all visualizations.
    """
    
    def __init__(self, df):
        """
        Initialize the class and automatically detect variable types.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to analyze
        """
        self.df = df
        self.var_num = df.select_dtypes(include=['number']).columns.tolist()
        self.var_bool = df.select_dtypes(include=['bool']).columns.tolist()
        self.var_cat = df.select_dtypes(include=['object']).columns.tolist()
        
        # Also check for string dtype explicitly
        self.var_cat.extend(df.select_dtypes(include=['string']).columns.tolist())
        self.var_cat = list(set(self.var_cat))  # Remove duplicates
        
        print(f"✓ Numerical variables ({len(self.var_num)}): {self.var_num}")
        print(f"✓ Boolean variables ({len(self.var_bool)}): {self.var_bool}")
        print(f"✓ Categorical variables ({len(self.var_cat)}): {self.var_cat}")
    
    def _fig_to_base64(self, fig):
        """
        Convert matplotlib figure to base64 string for embedding in HTML.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Matplotlib figure object
            
        Returns:
        --------
        str : Base64 encoded image string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64
    
    def _plotnine_to_base64(self, plot):
        """
        Convert plotnine plot to base64 string for embedding in HTML.
        
        Parameters:
        -----------
        plot : plotnine.ggplot
            plotnine plot object
            
        Returns:
        --------
        str : Base64 encoded image string
        """
        buffer = io.BytesIO()
        plot.save(buffer, format='png', dpi=100, verbose=False)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        return image_base64
    
    def _generate_table_rows(self):
        """
        Generate HTML table rows for columns and their data types.
        
        Returns:
        --------
        str : HTML table rows
        """
        rows = ""
        for col in self.df.columns:
            if col in self.var_num:
                type_class = "type-numerical"
                type_label = "Numerical"
            elif col in self.var_bool:
                type_class = "type-boolean"
                type_label = "Boolean"
            else:
                type_class = "type-categorical"
                type_label = "Categorical"
            
            rows += f"""
                                    <tr>
                                        <td><strong>{col}</strong></td>
                                        <td><span class="data-type {type_class}">{type_label}</span></td>
                                    </tr>
            """
        return rows
    
    def plot_num(self, explicativa=None):
        """
        Generate plots for numerical variables (boxplot and density).
        
        Parameters:
        -----------
        explicativa : str, optional
            Categorical variable for grouping comparisons
            
        Returns:
        --------
        list : List of base64 encoded plot images
        """
        plots_base64 = []
        
        for var in self.var_num:
            if explicativa and explicativa in self.df.columns:
                # Boxplot por categoría
                p1 = (
                    ggplot(self.df, aes(x=explicativa, y=var, fill=explicativa))
                    + geom_boxplot(alpha=0.6, outlier_color="maroon")
                    + theme_bw()
                    + labs(title=f"Boxplot {var} by {explicativa}", x=explicativa, y=var)
                )
                
                # Densidad por categoría
                p2 = (
                    ggplot(self.df, aes(x=var, fill=explicativa))
                    + geom_density(alpha=0.6)
                    + theme_bw()
                    + labs(title=f"Density {var} by {explicativa}", x=var, y="Density")
                )
            else:
                # Boxplot simple
                p1 = (
                    ggplot(self.df, aes(y=var))
                    + geom_boxplot(fill="#de3163", alpha=0.6)
                    + theme_bw()
                    + labs(title=f"Boxplot de {var}", y=var)
                )
                
                # Densidad simple
                p2 = (
                    ggplot(self.df, aes(x=var))
                    + geom_density(fill="#31deac", alpha=0.6)
                    + theme_bw()
                    + labs(title=f"Density de {var}", x=var, y="Density")
                )
            
            plots_base64.append(self._plotnine_to_base64(p1))
            plots_base64.append(self._plotnine_to_base64(p2))
        
        return plots_base64
    
    def plot_categorical(self):
        """
        Generate histograms for categorical variables.
        
        Returns:
        --------
        list : List of base64 encoded plot images
        """
        plots_base64 = []
        
        for var in self.var_cat:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            counts = self.df[var].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
            
            bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right')
            ax.set_xlabel(var, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'Histogram: {var}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
            
            plots_base64.append(self._fig_to_base64(fig))
        
        return plots_base64
    
    def plot_boolean(self):
        """
        Generate personalized histograms for boolean variables (2 classes only).
        
        Returns:
        --------
        list : List of base64 encoded plot images
        """
        plots_base64 = []
        
        for var in self.var_bool:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            counts = self.df[var].value_counts()
            colors = ['#FF6B6B', '#4ECDC4']  # Red for False, Teal for True
            labels = ['False', 'True']
            
            # Ensure proper ordering (False first, then True)
            ordered_counts = [counts.get(False, 0), counts.get(True, 0)]
            
            bars = ax.bar(labels, ordered_counts, color=colors, edgecolor='black', alpha=0.8, width=0.6)
            
            ax.set_xlabel(var, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'Boolean Distribution: {var}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels and percentages on bars
            total = sum(ordered_counts)
            for i, (bar, count) in enumerate(zip(bars, ordered_counts)):
                height = bar.get_height()
                percentage = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plots_base64.append(self._fig_to_base64(fig))
        
        return plots_base64
    
    def corr_matrix(self):
        """
        Generate correlation matrix heatmap for numerical variables.
        
        Returns:
        --------
        str : Base64 encoded correlation matrix image
        """
        if len(self.var_num) < 2:
            return None
        
        corr_data = self.df[self.var_num].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, cmap='viridis', fmt=".2f", 
                   cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Matrix - Numerical Variables", fontsize=14, fontweight='bold')
        
        return self._fig_to_base64(fig)
    
    def generate_html_report(self, filename='analysis_report.html', explicativa=None):
        """
        Generate comprehensive HTML report with all visualizations.
        
        Parameters:
        -----------
        filename : str, default='analysis_report.html'
            Output HTML filename
        explicativa : str, optional
            Categorical variable for grouping numerical comparisons
            
        Returns:
        --------
        str : Path to generated HTML file
        """
        # Generate table rows first
        table_rows = self._generate_table_rows()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                    overflow: hidden;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                
                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                
                .content {{
                    padding: 40px;
                }}
                
                .table-container {{
                    overflow-x: auto;
                    margin-bottom: 40px;
                }}
                
                .columns-table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                
                .columns-table thead {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                
                .columns-table th {{
                    padding: 15px 20px;
                    text-align: left;
                    font-weight: 600;
                    font-size: 1em;
                }}
                
                .columns-table td {{
                    padding: 12px 20px;
                    border-bottom: 1px solid #e9ecef;
                }}
                
                .columns-table tbody tr:last-child td {{
                    border-bottom: none;
                }}
                
                .columns-table tbody tr:hover {{
                    background-color: #f8f9fa;
                    transition: background-color 0.2s ease;
                }}
                
                .data-type {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 0.85em;
                    font-weight: 500;
                }}
                
                .type-numerical {{
                    background-color: #d4edff;
                    color: #0066cc;
                }}
                
                .type-categorical {{
                    background-color: #fff3cd;
                    color: #856404;
                }}
                
                .type-boolean {{
                    background-color: #d1e7dd;
                    color: #0f5132;
                }}
                
                .section {{
                    margin-bottom: 50px;
                }}
                
                .section h2 {{
                    color: #667eea;
                    font-size: 2em;
                    margin-bottom: 30px;
                    padding-bottom: 15px;
                    border-bottom: 3px solid #667eea;
                }}
                
                .plots-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                
                .plot-container {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                
                .plot-container img {{
                    width: 100%;
                    height: auto;
                    border-radius: 4px;
                    display: block;
                }}
                
                .full-width {{
                    grid-column: 1 / -1;
                }}
                
                .footer {{
                    background: #f8f9fa;
                    padding: 20px;
                    text-align: center;
                    color: #666;
                    border-top: 1px solid #e9ecef;
                    margin-top: 40px;
                }}
                
                .no-data {{
                    background: #fff3cd;
                    border-left: 5px solid #ffc107;
                    padding: 15px;
                    border-radius: 4px;
                    color: #856404;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Data Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="content">
                    <!-- Columns and Types Table Section -->
                    <div class="section">
                        <h2>📋 Columns & Data Types</h2>
                        <div class="table-container">
                            <table class="columns-table">
                                <thead>
                                    <tr>
                                        <th>Column Name</th>
                                        <th>Data Type</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {table_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
        """
        
        # Numerical Variables Section
        if self.var_num:
            html_content += """
                    <div class="section">
                        <h2>📈 Numerical Variables Analysis</h2>
                        <div class="plots-grid">
            """
            num_plots = self.plot_num(explicativa)
            for idx, plot_b64 in enumerate(num_plots):
                html_content += f"""
                            <div class="plot-container">
                                <img src="data:image/png;base64,{plot_b64}" alt="Numerical plot {idx}">
                            </div>
                """
            html_content += """
                        </div>
                    </div>
            """
        else:
            html_content += '<div class="no-data">No numerical variables found.</div>'
        
        # Correlation Matrix Section
        if len(self.var_num) >= 2:
            corr_plot = self.corr_matrix()
            html_content += f"""
                    <div class="section">
                        <h2>🔗 Correlation Analysis</h2>
                        <div class="plots-grid">
                            <div class="plot-container full-width">
                                <img src="data:image/png;base64,{corr_plot}" alt="Correlation matrix">
                            </div>
                        </div>
                    </div>
            """
        
        # Categorical Variables Section
        if self.var_cat:
            html_content += """
                    <div class="section">
                        <h2>🏷️ Categorical Variables Analysis</h2>
                        <div class="plots-grid">
            """
            cat_plots = self.plot_categorical()
            for idx, plot_b64 in enumerate(cat_plots):
                html_content += f"""
                            <div class="plot-container">
                                <img src="data:image/png;base64,{plot_b64}" alt="Categorical plot {idx}">
                            </div>
                """
            html_content += """
                        </div>
                    </div>
            """
        else:
            html_content += '<div class="no-data">No categorical variables found.</div>'
        
        # Boolean Variables Section
        if self.var_bool:
            html_content += """
                    <div class="section">
                        <h2>✓ Boolean Variables Analysis</h2>
                        <div class="plots-grid">
            """
            bool_plots = self.plot_boolean()
            for idx, plot_b64 in enumerate(bool_plots):
                html_content += f"""
                            <div class="plot-container">
                                <img src="data:image/png;base64,{plot_b64}" alt="Boolean plot {idx}">
                            </div>
                """
            html_content += """
                        </div>
                    </div>
            """
        else:
            html_content += '<div class="no-data">No boolean variables found.</div>'
        
        # Footer
        html_content += """
                    <div class="footer">
                        <p>📊 Analysis powered by graph_info | Data Analysis Suite</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report generated: {filename}")
        return filename
    
    def get_summary(self):
        """
        Get a summary of identified variable types.
        
        Returns:
        --------
        dict : Summary information
        """
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numerical_vars': self.var_num,
            'categorical_vars': self.var_cat,
            'boolean_vars': self.var_bool,
            'missing_values': self.df.isnull().sum().to_dict()
        }