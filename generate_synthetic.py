import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Load your real dataset
data = pd.read_csv('real_data.csv')

# STEP 1: Create Metadata and auto-detect types
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.save_to_json('metadata.json')


# STEP 2: Train the Synthesizer
model = CTGANSynthesizer(metadata)
model.fit(data)

# STEP 3: Generate synthetic data
synthetic_data = model.sample(100)

# STEP 4: Save it to CSV
synthetic_data.to_csv('synthetic_data.csv', index=False)
print("✅ Synthetic data saved to synthetic_data.csv")
