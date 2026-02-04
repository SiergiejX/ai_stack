#!/usr/bin/env python3
"""Generate 30 sample survey responses in JSON format."""

import json
import os
import random
from datetime import datetime, timedelta

# Define possible responses
GOALS = [
    "Pytanie o przedmiot/kurs",
    "Pomoc w projekcie/pracę zaliczeniową",
    "Informacje o zasadach zaliczenia",
    "Pytanie techniczne",
    "Inne"
]

DURATION = [
    "Mniej niż 5 minut",
    "5-15 minut",
    "15-30 minut",
    "Powyżej 30 minut"
]

PREFERENCES = [
    "Zdecydowanie wolę chat",
    "Raczej wolę chat",
    "Nie ma różnicy",
    "Raczej wolę konsultacje",
    "Zdecydowanie wolę konsultacje"
]

FUTURE_USE = [
    "Zdecydowanie tak",
    "Raczej tak",
    "Nie wiem",
    "Raczej nie",
    "Zdecydowanie nie"
]

CONTACT = ["Tak", "Nie"]

POSITIVE_FEEDBACK = [
    "Chat bardzo szybko udzielał odpowiedzi i był łatwy w obsłudze.",
    "Bardzo pomocny w wyjaśnianiu skomplikowanych tematów.",
    "Dostałem dokładne informacje, które bezpośrednio mi pomogły.",
    "Fajnie, że mogę pytać o wszystko bez obawy przed oceną.",
    "Szczególnie spodobało mi się, że chat dostosowywał odpowiedzi do mojego poziomu wiedzy.",
    "Szybkie odpowiedzi i zawsze dostępny - nie musiałem czekać na konsultacje.",
    "Bardzo naturalny sposób komunikacji, nie czułem się jak rozmawiam z robotem.",
    "Pomógł mi lepiej zrozumieć materiał."
]

IMPROVEMENTS = [
    "Chat mógłby czasami byś bardziej konkretny zamiast ogólnych wyjaśnień.",
    "Możliwość odwołania się do konkretnych stron w podręczniku byłaby przydatna.",
    "Czasami brakuje mi odsyłaczy do źródeł.",
    "Chat powinien wiedzieć więcej o specyficznych wymogach naszego kursu.",
    "Byłoby dobrze, gdyby chat mógł rysować diagramy lub wykresy.",
    "Chciałbym, żeby chat pamiętał poprzednie rozmowy.",
    "Chat mógłby bardziej nawiązywać do poprzednich odpowiedzi.",
    "Trudno mi czasami zrozumieć, czy odpowiedź chata jest oparta na aktualnych informacjach.",
    "Byłoby fajnie mieć możliwość weryfikacji odpowiedzi przez nauczyciela.",
    "Chat powinien znać więcej o zasadach uczelni."
]

PROBLEMS = [
    "Brak problemów, wszystko działało świetnie!",
    "Czasami chat nie rozumiewał mojego pytania i musiałem je powtórzyć.",
    "Jedna z odpowiedzi wydawała się sprzeczna z materiałem z wykładu.",
    "Brak problemów.",
    "Czasami chat generował zbyt długie odpowiedzi.",
    "Chat czasami odpowiadał poza tematem.",
    "Kilka razy chat nie wiedział, o co mi chodzi.",
]

ADDITIONAL_INFO = [
    "Chciałbym, żeby chat wiedział więcej o moim kierunku studiów.",
    "Chat powinien wiedzieć o terminach egzaminów i zadań.",
    "Byłoby przydatne, gdyby chat znał moje poprzednie notatki.",
    "Wolę, aby chat był neutralny wobec różnych punktów widzenia.",
    "Chat powinien znać ostatnie zmiany w programie nauczania.",
    "Brak dodatkowych informacji do dodania.",
    "Chat powinien być bardziej zaznajomiony z praktyką w branży.",
]

def generate_survey_response(response_id: int) -> dict:
    """Generate a single survey response."""
    # Random date in last 30 days
    days_ago = random.randint(0, 30)
    response_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
    
    # Generate Likert scale responses (1-5)
    def likert():
        return str(random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0])
    
    response = {
        "response_id": f"resp_{response_id:03d}",
        "timestamp": response_date,
        "student_id": f"student_{random.randint(10000, 99999)}",
        "student_email": f"student{random.randint(1000, 9999)}@uczelnia.edu.pl" if random.random() > 0.3 else None,
        "album_number": f"{random.randint(100000, 999999)}" if random.random() > 0.5 else None,
        
        # Basic Information
        "q1_goal": random.choice(GOALS),
        "q2_duration": random.choice(DURATION),
        
        # Satisfaction
        "q3_understanding": likert(),
        "q4_clarity": likert(),
        "q5_accuracy": likert(),
        "q6_usefulness": likert(),
        "q7_advanced_questions": likert(),
        
        # User Experience
        "q8_overall_experience": likert(),
        "q9_ease_of_use": likert(),
        "q10_response_time": likert(),
        "q11_ai_understanding": likert(),
        
        # Comparison
        "q12_chat_vs_teacher": random.choice(PREFERENCES),
        "q13_future_use": random.choice(FUTURE_USE),
        
        # Feedback
        "q14_positive": random.choice(POSITIVE_FEEDBACK),
        "q15_improvements": random.choice(IMPROVEMENTS),
        "q16_problems": random.choice(PROBLEMS),
        "q17_additional_info": random.choice(ADDITIONAL_INFO),
        
        # Contact
        "q18_contact_allowed": random.choice(CONTACT),
    }
    
    # Calculate satisfaction score (average of Likert questions)
    likert_scores = [
        int(response["q3_understanding"]),
        int(response["q4_clarity"]),
        int(response["q5_accuracy"]),
        int(response["q6_usefulness"]),
        int(response["q7_advanced_questions"]),
        int(response["q8_overall_experience"]),
        int(response["q9_ease_of_use"]),
        int(response["q10_response_time"]),
        int(response["q11_ai_understanding"]),
    ]
    response["satisfaction_score"] = round(sum(likert_scores) / len(likert_scores), 2)
    
    return response

def main():
    """Generate 30 survey responses and save to JSON files."""
    output_dir = os.path.join(os.path.dirname(__file__), 'responses')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 30 survey responses in {output_dir}...")
    
    all_responses = []
    
    for i in range(1, 31):
        response = generate_survey_response(i)
        all_responses.append(response)
        
        # Save individual response file
        filename = os.path.join(output_dir, f"survey_response_{i:02d}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        print(f"  Created: survey_response_{i:02d}.json (satisfaction: {response['satisfaction_score']}/5)")
    
    # Save summary file with all responses
    summary_file = os.path.join(output_dir, "all_responses_summary.json")
    summary = {
        "total_responses": len(all_responses),
        "average_satisfaction": round(sum(r['satisfaction_score'] for r in all_responses) / len(all_responses), 2),
        "generated_at": datetime.now().isoformat(),
        "responses": all_responses
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Generated 30 survey responses!")
    print(f"   Average satisfaction score: {summary['average_satisfaction']}/5")
    print(f"   Summary file: all_responses_summary.json")

if __name__ == '__main__':
    main()
