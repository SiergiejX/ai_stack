# Survey Management Module
## Agent 3 - Analytics

Ten katalog zawiera wszystkie pliki związane z ankietami studentów i zarządzaniem danymi w Qdrancie.

### Struktura katalogów

- `responses/` - Katalog z zapisanymi odpowiedziami ankiet (30 plików survey_response_XX.json)
- `student_satisfaction_survey.html` - Formularz ankiety w HTML
- `student_satisfaction_survey.json` - Schemat ankiety w formacie JSON
- `qdrant_setup_simple.py` - Skrypt do inicjalizacji kolekcji Qdranta
- `qdrant_setup.py` - Alternatywny skrypt z wsparciem dla AI embeddings
- `generate_responses.py` - Skrypt do generowania przykładowych odpowiedzi ankiet

### Integracja z Agent 3 Analytics

Agent 3 Analytics ma dostęp do tego modułu i może:
- Przesyłać dane ankiet do Qdranta
- Wyszukiwać podobne ankiety
- Analizować trendy w opiniach studentów
- Generować raporty z wyników ankiet

### Użycie

#### Inicjalizacja kolekcji Qdranta

```bash
cd /path/to/agent3_analytics/surveys
python qdrant_setup_simple.py
```

#### Generowanie danych testowych

```bash
python generate_responses.py
```

### Schemat danych ankiety

Każda ankieta zawiera:
- `response_id` - Unikalny identyfikator
- `student_id` - ID studenta
- `timestamp` - Data/czas odpowiedzi
- `satisfaction_score` - Ogólny wynik zadowolenia (1-5)
- `q1_goal` - Cel nauczania
- `q2_duration` - Czas potrzebny
- `q3_understanding` - Zrozumienie pytań (1-5)
- `q4_clarity` - Jasność odpowiedzi (1-5)
- `q5_accuracy` - Dokładność (1-5)
- `q6_usefulness` - Użyteczność (1-5)
- `q7_advanced_questions` - Zaawansowane pytania (1-5)
- `q8_overall_experience` - Ogólne doświadczenie (1-5)
- `q9_ease_of_use` - Łatwość użycia (1-5)
- `q10_response_time` - Szybkość odpowiedzi (1-5)
- `q11_ai_understanding` - Zrozumienie AI (1-5)
- `q12_chat_vs_teacher` - Chat vs nauczyciel
- `q13_future_use` - Przyszłe użycie
- `q14_positive` - Pozytywne aspekty
- `q15_improvements` - Ulepszenia
- `q16_problems` - Problemy napotkane

### Kolekcja Qdrant

- **Nazwa**: `survey_responses`
- **Wymiar wektora**: 128
- **Metryka odległości**: COSINE
- **Liczba punktów**: 30 (odpowiedzi ankiet)
