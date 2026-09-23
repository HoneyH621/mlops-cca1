import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load the student dataset
data = pd.read_csv("data/student_data.csv")

# Split the dataset into reference and current data
reference_data = data.iloc[:10]
current_data = data.iloc[10:]

# Create a data drift report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Run the report
snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data
)

# Save the monitoring report
snapshot.save_html("evidently_report.html")

print("Evidently monitoring report generated successfully.")