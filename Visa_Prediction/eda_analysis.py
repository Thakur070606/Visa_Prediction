import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VisaEDA:
    def __init__(self, df):
        self.df = df
        plt.style.use('seaborn-v0_8')
        
    def target_distribution(self):
        """Analyze target variable distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        self.df['case_status'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Visa Case Status Distribution')
        ax1.set_xlabel('Case Status')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Pie chart
        self.df['case_status'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                                                  colors=['skyblue', 'lightcoral'])
        ax2.set_title('Visa Case Status Percentage')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def continent_analysis(self):
        """Analyze visa approval by continent"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count by continent
        continent_counts = self.df['continent'].value_counts()
        continent_counts.plot(kind='bar', ax=ax1, color='lightgreen')
        ax1.set_title('Applications by Continent')
        ax1.set_xlabel('Continent')
        ax1.set_ylabel('Number of Applications')
        ax1.tick_params(axis='x', rotation=45)
        
        # Approval rate by continent
        approval_rate = self.df.groupby('continent')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        ).sort_values(ascending=False)
        
        approval_rate.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Visa Approval Rate by Continent')
        ax2.set_xlabel('Continent')
        ax2.set_ylabel('Approval Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('continent_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def education_analysis(self):
        """Analyze visa approval by education level"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Education distribution
        education_counts = self.df['education_of_employee'].value_counts()
        education_counts.plot(kind='bar', ax=ax1, color='purple', alpha=0.7)
        ax1.set_title('Applications by Education Level')
        ax1.set_xlabel('Education Level')
        ax1.set_ylabel('Number of Applications')
        ax1.tick_params(axis='x', rotation=45)
        
        # Approval rate by education
        edu_approval = self.df.groupby('education_of_employee')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        ).sort_values(ascending=False)
        
        edu_approval.plot(kind='bar', ax=ax2, color='teal')
        ax2.set_title('Visa Approval Rate by Education Level')
        ax2.set_xlabel('Education Level')
        ax2.set_ylabel('Approval Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('education_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def experience_analysis(self):
        """Analyze impact of job experience and training"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Job experience impact
        exp_approval = self.df.groupby('has_job_experience')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        exp_approval.plot(kind='bar', ax=ax1, color=['red', 'green'])
        ax1.set_title('Approval Rate by Job Experience')
        ax1.set_xlabel('Has Job Experience')
        ax1.set_ylabel('Approval Rate (%)')
        ax1.tick_params(axis='x', rotation=0)
        
        # Training requirement impact
        training_approval = self.df.groupby('requires_job_training')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        training_approval.plot(kind='bar', ax=ax2, color=['blue', 'orange'])
        ax2.set_title('Approval Rate by Training Requirement')
        ax2.set_xlabel('Requires Job Training')
        ax2.set_ylabel('Approval Rate (%)')
        ax2.tick_params(axis='x', rotation=0)
        
        # Full-time position impact
        fulltime_approval = self.df.groupby('full_time_position')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        fulltime_approval.plot(kind='bar', ax=ax3, color=['purple', 'yellow'])
        ax3.set_title('Approval Rate by Full-time Position')
        ax3.set_xlabel('Full-time Position')
        ax3.set_ylabel('Approval Rate (%)')
        ax3.tick_params(axis='x', rotation=0)
        
        # Region analysis
        region_approval = self.df.groupby('region_of_employment')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        ).sort_values(ascending=False)
        region_approval.plot(kind='bar', ax=ax4, color='brown')
        ax4.set_title('Approval Rate by Region')
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Approval Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('experience_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def wage_analysis(self):
        """Analyze wage patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Wage distribution by status
        certified_wages = self.df[self.df['case_status'] == 'Certified']['prevailing_wage']
        denied_wages = self.df[self.df['case_status'] == 'Denied']['prevailing_wage']
        
        ax1.hist([certified_wages, denied_wages], bins=50, alpha=0.7, 
                label=['Certified', 'Denied'], color=['green', 'red'])
        ax1.set_title('Wage Distribution by Case Status')
        ax1.set_xlabel('Prevailing Wage')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_xlim(0, 200000)  # Limit x-axis for better visualization
        
        # Box plot of wages by status
        self.df.boxplot(column='prevailing_wage', by='case_status', ax=ax2)
        ax2.set_title('Wage Distribution by Case Status (Box Plot)')
        ax2.set_xlabel('Case Status')
        ax2.set_ylabel('Prevailing Wage')
        ax2.set_ylim(0, 200000)
        
        # Wage unit analysis
        wage_unit_approval = self.df.groupby('unit_of_wage')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        wage_unit_approval.plot(kind='bar', ax=ax3, color='cyan')
        ax3.set_title('Approval Rate by Wage Unit')
        ax3.set_xlabel('Unit of Wage')
        ax3.set_ylabel('Approval Rate (%)')
        ax3.tick_params(axis='x', rotation=0)
        
        # Company size vs approval (if company_age exists)
        if 'company_age' in self.df.columns:
            # Create company age bins
            self.df['age_bin'] = pd.cut(self.df['company_age'], 
                                      bins=[0, 10, 25, 50, float('inf')],
                                      labels=['New (0-10)', 'Young (11-25)', 
                                             'Mature (26-50)', 'Established (50+)'])
            
            age_approval = self.df.groupby('age_bin')['case_status'].apply(
                lambda x: (x == 'Certified').mean() * 100
            )
            age_approval.plot(kind='bar', ax=ax4, color='magenta')
            ax4.set_title('Approval Rate by Company Age')
            ax4.set_xlabel('Company Age')
            ax4.set_ylabel('Approval Rate (%)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('wage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[numerical_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Variables')
            plt.tight_layout()
            plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    def generate_summary_stats(self):
        """Generate summary statistics"""
        print("=== VISA PREDICTION DATASET SUMMARY ===")
        print(f"Total Applications: {len(self.df):,}")
        print(f"Certified Applications: {len(self.df[self.df['case_status'] == 'Certified']):,}")
        print(f"Denied Applications: {len(self.df[self.df['case_status'] == 'Denied']):,}")
        print(f"Overall Approval Rate: {(self.df['case_status'] == 'Certified').mean()*100:.1f}%")
        
        print("\n=== TOP INSIGHTS ===")
        
        # Top continent by applications
        top_continent = self.df['continent'].value_counts().index[0]
        print(f"Most applications from: {top_continent}")
        
        # Best approval rate by continent
        continent_approval = self.df.groupby('continent')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        best_continent = continent_approval.idxmax()
        print(f"Highest approval rate continent: {best_continent} ({continent_approval[best_continent]:.1f}%)")
        
        # Education impact
        edu_approval = self.df.groupby('education_of_employee')['case_status'].apply(
            lambda x: (x == 'Certified').mean() * 100
        )
        best_education = edu_approval.idxmax()
        print(f"Best education level for approval: {best_education} ({edu_approval[best_education]:.1f}%)")
        
        # Experience impact
        exp_yes = self.df[self.df['has_job_experience'] == 'Y']['case_status'].apply(lambda x: x == 'Certified').mean() * 100
        exp_no = self.df[self.df['has_job_experience'] == 'N']['case_status'].apply(lambda x: x == 'Certified').mean() * 100
        print(f"Approval rate with experience: {exp_yes:.1f}%")
        print(f"Approval rate without experience: {exp_no:.1f}%")
        
    def run_complete_eda(self):
        """Run complete EDA analysis"""
        print("Starting Exploratory Data Analysis...")
        
        self.generate_summary_stats()
        self.target_distribution()
        self.continent_analysis()
        self.education_analysis()
        self.experience_analysis()
        self.wage_analysis()
        self.correlation_analysis()
        
        print("\nEDA completed! All visualizations saved as PNG files.")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Visa_Predection_Dataset.csv')
    
    # Run EDA
    eda = VisaEDA(df)
    eda.run_complete_eda()