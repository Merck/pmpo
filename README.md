# Probabilistic Multi-Parameter Optimization (pMPO)

An implementation of the pMPO method described in:

**Hakan Gunaydin**. Probabilistic Approach to Generating MPOs and Its Application as a Scoring Function for CNS Drugs.
ACS Med Chem Lett 7, 89-93 (2016)

Used to create MPO models with high levels of discrimination between good and bad samples using an independent t-test
and removal of linearly correlated descriptors.

## Author

Scott Arne Johnson <scott.johnson6@merck.com>

Rewritten and tested against the original source code used in the paper by Hakan Gunaydin.

## Usage

The pMPO model calculations are built around Pandas DataFrames. I'll expand this documentation later to describe how
to do that for molecules with the *oenotebook* package.

### Simple Usage

This will show building a model with the pickled Pandas DataFrame ``pMPO/test/assets/CNS_MPO.df.pkl`` included in the 
testing suite for this project. This is a DataFrame created from the data used in the original publication.

**If you do not know how to build a DataFrame like this from molecule data, the a tutorial will be added in the near 
future**

```text
import pandas as pd
df = pd.read_pickle('pMPO/tests/assets/CNS_MPO.df.pkl')
```

Once you've created your Pandas DataFrame with your molecule data. You can use the ``pMPOBuilder`` class to create your 
model. The score for each descriptor in a pMPO is a weighted Gaussian function multipled by a sigmoidal function term 
that further biases against bad compounds. The published CNS pMPO did not include that additional bias. It is controlled
by the additional parameter ```sigmoidal_correction```, which is ```True``` by default.

```python
builder = pMPOBuilder(df, good_column='CNS', model_name='CNS pMPO', sigmoidal_correction=False)
```

That's it! You've re-created the CNS pMPO. We can get a usable model from the following:

```python
model = builder.get_pMPO()
```

We can inspect the model by just printing it as a string ```print(model)```

```text
CNS pMPO: [CLOGD_ACD_V15] 0.13 * np.exp(-1.0 * (x - 1.81)^2 / (2.0 * (1.93)^2)) + [HBD] 0.27 * np.exp(-1.0 * 
(x - 1.09)^2 / (2.0 * (0.89)^2)) + [MBPKA] 0.12 * np.exp(-1.0 * (x - 8.07)^2 / (2.0 * (2.21)^2)) + [MW] 0.16 * 
np.exp(-1.0 * (x - 304.70)^2 / (2.0 * (94.05)^2)) + [TPSA] 0.33 * np.exp(-1.0 * (x - 50.70)^2 / (2.0 * (28.30)^2))
```

Note that the properties are in all caps because by default ```case_insensitive=True``` for model properties. If we 
wanted to build the same model using the default sigmoidal correction:

```python
builder = pMPOBuilder(df, good_column='CNS', model_name='CNS pMPO with Correction')
model_with_correction = builder.get_pMPO()
```

And inspecting this model shows additional terms:

```text
CNS pMPO with Correction: [CLOGD_ACD_V15] 0.13 * np.exp(-1.0 * (x - 1.81)^2 / (2.0 * (1.93)^2)) * np.power(1.0 + 0.02 * 
np.power(131996.99, -1.0 * (x - 1.81)), -1.0) + [HBD] 0.27 * np.exp(-1.0 * (x - 1.09)^2 / (2.0 * (0.89)^2)) * 
np.power(1.0 + 0.09 * np.power(0.00, -1.0 * (x - 1.09)), -1.0) + [MBPKA] 0.12 * np.exp(-1.0 * (x - 8.07)^2 / (2.0 * 
(2.21)^2)) * np.power(1.0 + 0.02 * np.power(1459310.78, -1.0 * (x - 8.07)), -1.0) + [MW] 0.16 * np.exp(-1.0 * 
(x - 304.70)^2 / (2.0 * (94.05)^2)) * np.power(1.0 + 0.03 * np.power(0.83, -1.0 * (x - 304.70)), -1.0) + [TPSA] 0.33 * 
np.exp(-1.0 * (x - 50.70)^2 / (2.0 * (28.30)^2)) * np.power(1.0 + 0.15 * np.power(0.79, -1.0 * (x - 50.70)), -1.0)
```

You can see how it has the name of the model ("CNS pMPO") followed by all the relevant data that would be expected on
input to the model (e.g. "MW") and the equation for that particular piece of data.

You can use your model on any dictionary of data with the expected tags. For example:

```python
# A new observation wih data
# Data that is missing is given a score of 0.0 and irrelevant data is not used
abacavir = {
    'TPSA': 101.88,
    'HBA': 6,
    'HBD': 3,
    'MW': 286.33231,
    'cLogD_ACD_v15': 0.72000003,
    'mbpKa': 6.5300002,
    'cLogP_ACD_v15': 0.72000003     
}

score = model(**abacavir)

print(score)
> 0.44567876450073463
```

### Model Analytics

You can get all the analytics to assess the model you just built.

```python
stats = builder.get_pMPO_statistics()
```

This will return a Pandas DataFrame with the following columns (for each descriptor column, e.g. "TPSA"):

```text
name: The name of the descriptor (e.g. 'TPSA')
good_mean: The mean of the "good" samples for this descriptor
good_std: The standard deviation of the "good" samples
good_nsamples: The number of "good" samples with this data
bad_mean: The mean of the "bad" observations
bad_std: The standard deviation of the "bad" samples
bad_nsamples: The number of "bad" samples with this data
p_value: The p-value from the independent t-test between good and bad samples
significant: Whether the p-value was in the significance threshold set on model building
cutoff: The cutoff calculated for this descriptor
inflection: The y-inflection point calculated during the sigmoidal function fitting
b: The 'b' parameter for the sigmoidal function (see Hakan's paper)
c: The 'c' parameter for the sigmoidal function (see Hakan's paper)
z: The z-score for this descriptor
w: The weight calculated for this descriptor from the z-score
selected: Whether this descriptor was selected to be included in the pMPO
```

You can also ask for the NxN descriptor correlation matrix as a Pandas DataFrame. The matrix is populated by r^2 values 
for the linear correlations of each descriptor pair (square of Pearson's r).

```python
stats = builder.get_descriptor_correlation()
```

### Advanced Usage

You can customize the statistical cutoffs when building models as well.

```text
pMPOBuilder(
    df: (str) The input DataFrame [Required]
    good_column: (str) The name of the column that defines good/bad observations [Required]
    good_value: (str) The value of a good observation in the good_column [default: Assortment of values equal to TRUE]
    pMPO_good_column_name: (str) The boolean TRUE/FALSE interpretation of good_column [default: pMPO_POSITIVE]
    min_samples: (int) Minimum number of samples with good or bad data to calculate p-value statistics [default: 10]
    p_cutoff: (float) p-value cutoff to determine significant separation between good and bad molecules [default: 0.01]
    q_cutoff: (float) The q-value cutoff used in parameterizing the sigmoidal functions [default: 0.05]
    r2_cutoff: (float) The r^2 cutoff for determining linearly correlated descriptors [default: 0.53]
)
```

Models are simple and can be pickled for storage and re-use:

```python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
```

You can load it back up again and use it right away.

## License

Copyright 2017 Merck & Co., Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.