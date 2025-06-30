<p align="center"><h1 align="center">APPLYBN</h1></p>
<p align="center">
	<img src="https://img.shields.io/github/license/Anaxagor/applybn?style=default&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
</p>
<p align="center">Создано с помощью инструментов и технологий:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white"alt="Python">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=default&logo=GitHub-Actions&logoColor=white"alt="GitHub%20Actions">
</p>
<br>

applybn — это многоцелевой фреймворк с открытым исходным кодом, основанный на байесовских сетях и каузальных моделях.
Он представляет анализ данных, основанный на понятных и интерпретируемых алгоритмах байесовских сетей и каузальных моделей.
![image](https://github.com/user-attachments/assets/996f8e5a-1742-4849-a64f-58b97a4cf17d)

## Ключевые особенности
### 1. **Обнаружение аномалий во временных рядах и табличных данных**
#### **Табличные данные**
   - Метод сочетает в себе обнаружение на основе условных распределений байесовской сети и оценку на основе близости.
   - Метод позволяет выявлять как выбросы по плотности, так и выбросы, возникающие из-за нарушения зависимостей между признаками.
#### **Временные ряды**
   - Метод выявляет выбросы в многомерных временных рядах, используя динамические байесовские сети (DBN) для моделирования временных и межпеременных зависимостей.
   - Его ключевые преимущества заключаются в улавливании сложных временных взаимозависимостей, автоматизации классификации выбросов с помощью адаптивных стратегий анализа оценок.
### 2. **Генерация синтетических данных**
   - Фреймворк включает методы для генерации синтетических обучающих данных при обнаружении дисбаланса классов в наборе данных. Используя гибридные байесовские сети (с моделями гауссовых смесей), он генерирует сбалансированные синтетические данные, улучшая результаты обучения моделей.
   - Метод позволяет учитывать взаимодействия между переменными, что повышает качество синтетических данных.

### 3. **Отбор и извлечение признаков**
#### **Отбор признаков на основе причинно-следственного эффекта**
   - Этот метод количественно оценивает причинно-следственные эффекты с помощью теории информации и пост-нелинейных моделей, автоматически выбирая признаки с ненулевым причинным влиянием на ключевые показатели эффективности (KPI) путем анализа снижения неопределенности в энтропии.
   - Метод отличается интерпретируемостью и стабильностью, отдавая приоритет причинно-следственным связям, избегает ручной настройки пороговых значений и надежно работает даже с ограниченными или нелинейными промышленными данными, повышая надежность программных датчиков.
#### **Отбор признаков на основе нормализованной взаимной информации**
   - Метод сочетает отбор признаков с правилами Мика для выявления марковских покрытий, используя взаимную информацию для сохранения узлов со значительными зависимостями и отсеивания нерелевантных или избыточных признаков в байесовских сетях.
   - Его ключевые преимущества включают эффективную обработку высокоразмерных данных за счет сосредоточения на локальных структурах, повышенную точность по сравнению с традиционными методами.
#### **Извлечение признаков на основе байесовских сетей**
   - Метод работает путем вывода условных зависимостей между признаками с использованием байесовских сетей и добавления вероятностных параметров (λ), представляющих эти зависимости, в исходный набор данных.
   - Его преимущества включают повышенную точность классификации в различных областях, снижение зависимости от априорных знаний и низкую вычислительную сложность, что обеспечивает надежное и интерпретируемое обогащение признаков без ограничений, специфичных для предметной области.

### 4. **Объяснимый анализ**
#### **Причинно-следственный анализ для моделей машинного обучения**
   - Метод анализа компонентов модели - строится структурная каузальная модель (SCM) для анализа моделей глубокого обучения, что позволяет отсеивать неважные части (например, фильтры в CNN) путем оценки их причинной важности.
   - Метод объяснения влияния данных на прогнозы позволяет делать причинно-следственные выводы между признаками и показателями уверенности модели. Рассчитывая **средний причинный эффект (ACE)**, он помогает определить, какие признаки значительно влияют на неопределенность модели, предоставляя ценную информацию для улучшения или отладки моделей.

### 5. **Совместимость со Scikit-learn**
Все оценщики и преобразователи данных совместимы со scikit-learn.

## Установка

Чтобы начать работу с applybn, клонируйте репозиторий и установите необходимые зависимости.

```bash
git clone https://github.com/Anaxagor/applybn.git
cd applybn
poetry install
```

## Пример конвейера Scikit-learn

Пример, демонстрирующий, как использовать несколько компонентов из библиотеки `applybn` в Pipeline, совместимом со scikit-learn. Мы рассмотрим отбор признаков, генерацию признаков, ресемплирование для несбалансированных наборов данных и обнаружение аномалий.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator
from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector


def generate_example_data(n_samples=200, n_features=5, n_cat_features=2, target_name='target_class', imbalance_ratio=0.1, random_state=42):
    """Генерирует синтетический несбалансированный набор данных."""
    rng = np.random.RandomState(random_state)
    X_cont_data = rng.rand(n_samples, n_features - n_cat_features)
    cont_feature_names = [f'cont_feature_{j}' for j in range(n_features - n_cat_features)]
    X_cont_df = pd.DataFrame(X_cont_data, columns=cont_feature_names)
    
    X_cat_df = pd.DataFrame()
    for i in range(n_cat_features):
        cat_feature_name = f'cat_feature_{i}'
        temp_cont_for_cat = rng.rand(n_samples, 1)
        n_bins_cat = rng.randint(2, 4)
        discretizer = KBinsDiscretizer(n_bins=n_bins_cat, encode='ordinal', strategy='uniform', subsample=None, random_state=rng)
        X_cat_df[cat_feature_name] = discretizer.fit_transform(temp_cont_for_cat).astype(int).astype(str)
        
    X_df = pd.concat([X_cont_df, X_cat_df], axis=1)
    
    n_class1 = int(n_samples * imbalance_ratio)
    n_class0 = n_samples - n_class1
    y_array = np.array([0] * n_class0 + [1] * n_class1)
    rng.shuffle(y_array)
    y_series = pd.Series(y_array, name=target_name)
    return X_df, y_series

# --- Основной пример ---
TARGET_NAME = 'target_class_example'

# 1. Генерация данных
print("1. Генерация синтетических данных...")
X_df, y_s = generate_example_data(n_samples=250, n_features=6, n_cat_features=2, target_name=TARGET_NAME, imbalance_ratio=0.15, random_state=42)
X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(X_df, y_s, test_size=0.3, random_state=42, stratify=y_s)
print(f"   Обучающие данные: {X_train_df.shape}, Тестовые данные: {X_test_df.shape}")
print(f"   Распределение цели (обучение):\\n{y_train_s.value_counts(normalize=True).to_string()}\n")

# 2. Причинно-следственный отбор признаков
print("2. Причинно-следственный отбор признаков...")
selector = CausalFeatureSelector(n_bins=3) # n_bins=3 для небольших данных
selector.fit(X_train_df.to_numpy(), y_train_s.to_numpy())
selected_features_mask = selector.get_support()

if not np.any(selected_features_mask):
    print("   Предупреждение: CausalFeatureSelector не выбрал ни одного признака. Используются все признаки.")
    X_train_selected_df = X_train_df.copy()
    X_test_selected_df = X_test_df.copy()
else:
    X_train_selected_df = X_train_df.loc[:, selected_features_mask]
    X_test_selected_df = X_test_df.loc[:, selected_features_mask]
print(f"   Выбрано {len(X_train_selected_df.columns)} признаков: {X_train_selected_df.columns.tolist()}\n")

# 3. Конвейер обработки (Генерация признаков, Передискретизация, Классификация)
print("3. Конвейер обработки (BNFeatureGenerator, BNOverSampler, Classifier)...")
X_train_selected_df.columns = X_train_selected_df.columns.astype(str)
X_test_selected_df.columns = X_test_selected_df.columns.astype(str)

processing_pipeline = ImbPipeline([
    ('bn_feature_generator', BNFeatureGenerator()), 
    ('oversampler', BNOverSampler(class_column=TARGET_NAME, strategy='minority', shuffle=True)),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

processing_pipeline.fit(X_train_selected_df, y_train_s)
pipeline_score = processing_pipeline.score(X_test_selected_df, y_test_s)
print(f"   Точность конвейера на тесте: {pipeline_score:.4f}\n")

# 4. Пример TabularDetector
print("4. Обнаружение аномалий с помощью TabularDetector...")
# TabularDetector ожидает target_name в входном DataFrame X.
X_train_for_detector = X_train_selected_df.copy()
X_train_for_detector[TARGET_NAME] = y_train_s

detector = TabularDetector(target_name=TARGET_NAME, verbose=0) # verbose=0 для меньшего вывода
detector.fit(X_train_for_detector) # y = None, так как цель находится в X
anomaly_predictions_train = detector.predict(X_train_for_detector)
print(f"   Аномалии в обучающих данных: {np.sum(anomaly_predictions_train)}/{len(anomaly_predictions_train)}")

X_test_for_detector = X_test_selected_df.copy()
X_test_for_detector[TARGET_NAME] = y_test_s
test_anomaly_predictions = detector.predict(X_test_for_detector)
print(f"   Аномалии в тестовых данных: {np.sum(test_anomaly_predictions)}/{len(test_anomaly_predictions)}\n")

print("--- Пример завершен ---")
```

## Помощь и поддержка

### Документация с примерами

Документацию с примерами можно найти [здесь](https://anaxagor.github.io/applybn/).

### Связь

Не стесняйтесь создавать новые [задачи](https://github.com/Anaxagor/applybn/issues).

Если у вас есть вопросы или предложения, вы можете связаться с нами по следующим ресурсам:
* [Helpdesk в чате Telegram](https://t.me/+4FOcyF0Rri00ZGEy)
* ideeva@itmo.ru (Ирина Деева)

### Участие в разработке

Вклад в applybn приветствуется! Если вы заинтересованы в улучшении каких-либо функций или тестировании новых веток, пожалуйста, ознакомьтесь с разделом [Как внести вклад](https://anaxagor.github.io/applybn/development/contributing/) документации.

### Лицензия

applybn распространяется под лицензией MIT. Дополнительную информацию см. в файле `LICENSE`.

## Цитирование

```bibtex
@software{applybn,
  author = {Irina Deeva},
  title = {applybn},
  url = {https://github.com/Anaxagor/applybn},
  version = {0.1.0},
  date = {2025-05-03},
}
```

## Благодарности
Проект поддержан [Фондом содействия инновациям](https://fasie.ru/). 