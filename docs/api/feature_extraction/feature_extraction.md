# Генерация признаков с помощью байесовских сетей

[Ссылка на руководство пользователя](../../user-guide/feature_extraction/feature_extraction.md)

::: applybn.feature_extraction.bn_feature_extractor.BNFeatureGenerator

# Пример

```python
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator
import pandas as pd

data = pd.read_csv('your_data.csv')

generator = BNFeatureGenerator()
generator.fit(data)

data_with_features = generator.transform(data)
```
