# Instrukcja Dostępu do Ankiety

## 🌐 Dostęp do Formularza Ankiety

Formularz ankiety jest dostępny pod następującymi adresami:

### Lokalnie (jeśli uruchamiasz na localhost)
- **Bezpośredni dostęp**: http://localhost:8003/surveys
- **Plik HTML**: http://localhost:8003/static/survey.html

### Z innej maszyny w sieci
- Zastąp `localhost` adresem IP lub domeną serwera, np:
  - http://192.168.1.100:8003/surveys
  - http://your-server.com:8003/surveys

## 📝 Jak Wypełnić Ankietę

1. Otwórz przeglądarkę internetową
2. Wejdź na adres http://localhost:8003/surveys
3. Wypełnij wszystkie pytania oznaczone gwiazdką (*)
4. Odpowiadaj szczerze na pytania dotyczące Twojego doświadczenia
5. Kliknij przycisk **"Wyślij ankietę"**

## ✅ Co Się Dzieje Po Wysłaniu

Gdy klikniesz "Wyślij ankietę":

1. **Ankieta jest wysyłana do API** (`/api/surveys`)
2. **Embedding jest generowany** - ankieta jest konwertowana na wektor dla Qdranta
3. **Dane są zapisywane w Qdrancie** - w kolekcji `survey_responses`
4. **Plik JSON jest zapisywany** - w katalogu `agent3_analytics/surveys/responses/`
5. **Potwierdzenie jest wyświetlane** - pokazuje się komunikat z ID ankiety

## 📊 Struktura Zapisywanych Danych

Każda ankieta zawiera:

```json
{
  "response_id": "resp_001",
  "timestamp": "2026-01-30T14:00:00Z",
  "student_id": "student_12345",
  "satisfaction_score": 4.5,
  "q1_goal": "Cel rozmowy...",
  "q2_duration": "Czas trwania...",
  "q3_understanding": "5",
  ...
  "q16_problems": "Napotkane problemy..."
}
```

## 🗄️ Gdzie Są Przechowywane Ankiety

### Qdrant (Wektorowa Baza Danych)
- **Lokacja**: Kontener Qdrant (port 6333)
- **Kolekcja**: `survey_responses`
- **Wymiar wektora**: 128
- **Liczba ankiet**: Wyświetlona na `/surveys/stats`

### Pliki JSON
- **Lokacja**: `agent3_analytics/surveys/responses/`
- **Format**: `{response_id}.json`
- **Przykład**: `resp_001.json`, `test_file_save_001.json`

## 🔍 Sprawdzenie Statystyk

Aby zobaczyć ile ankiet jest w systemie, otwórz:
```
http://localhost:8003/surveys/stats
```

Odpowiedź:
```json
{
  "collection": "survey_responses",
  "points_count": 35,
  "status": "ready"
}
```

## 🐛 Rozwiązywanie Problemów

### Problem: "Błąd wysyłania do serwera"
- **Przyczyna**: Przeglądarka nie może się połączyć z API
- **Rozwiązanie**: 
  - Upewnij się, że kontener agent3_analytics jest uruchomiony: `docker ps | grep agent3`
  - Sprawdź czy API jest dostępne: `http://localhost:8003/surveys/stats`
  - Jeśli uruchamiasz z innej maszyny, użyj prawidłowego adresu IP zamiast localhost

### Problem: "Ankieta zapisana lokalnie" (plik pobrany, ale nie wysłana do Qdranta)
- **Przyczyna**: API nie odpowiada
- **Rozwiązanie**: Sprawdź logi kontenera
  ```bash
  docker logs agent3_analytics
  ```

### Problem: Plik nie pojawia się w responses/
- **Przyczyna**: Może być problem z uprawnieniami
- **Rozwiązanie**: Sprawdź zawartość katalogu
  ```bash
  ls -la agent3_analytics/surveys/responses/
  ```

## 📞 Obsługiwane Typ Ankiet

### Sekcje Ankiety

1. **Informacje Podstawowe**
   - Cel rozmowy
   - Czas trwania

2. **Ocena Zadowolenia** (Skala Likerta 1-5)
   - Zrozumienie pytań
   - Jasność odpowiedzi
   - Dokładność
   - Użyteczność
   - Zaawansowane pytania

3. **Doświadczenie Użytkownika** (Skala Likerta 1-5)
   - Ogólne doświadczenie
   - Łatwość użycia
   - Czas odpowiedzi
   - Zrozumienie AI

4. **Porównanie z Alternatywami**
   - Chat vs. konsultacje nauczyciela
   - Przyszłe użycie

5. **Opinie i Sugestie**
   - Co się spodobało?
   - Co mogłoby być lepsze?
   - Problemy napotkane
   - Dodatkowe informacje

6. **Dane Kontaktowe** (Opcjonalne)
   - Email
   - Numer albumu
   - Zgoda na kontakt

## 🔐 Bezpieczeństwo

- Ankiety są anonimowe (student_id jest generowany losowo)
- Email i nr albumu są opcjonalne
- Wszystkie dane są szyfrowane w Qdrancie
- API ma włączony CORS dla bezpiecznej komunikacji

## 📈 Analiza Danych

Po zebraniu ankiet, możesz:

1. Wyszukiwać podobne ankiety w Qdrancie
2. Analizować średnią ocenę zadowolenia
3. Ekstrakcję trendów z opinii tekstowych
4. Generować raporty za pomocą Agent 3 Analytics

Aby uzyskać dostęp do wyszukiwarki, użyj metody `SurveyQdrantManager` z modułu surveys.
