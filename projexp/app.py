from flask import Flask, render_template, request, send_from_directory
import json
import random
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Ensure you have a directory for downloads
DOWNLOAD_FOLDER = r'E:\projexp\downloads'  # Updated path to the downloads directory
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Class to handle Timetable Optimization
class TimetableOptimizer:
    def __init__(self, data, sections, days, periods):
        self.data = data
        self.sections = sections
        self.days = days
        self.periods = periods
        self.day_mapping = {day: idx for idx, day in enumerate(days)}
        self.gradient_boosting_model, self.feature_columns = self.train_gradient_boosting_model()
        self.naive_bayes_model = self.train_naive_bayes_model()
        self.bert_model, self.tokenizer = self.load_bert_model()

    def train_gradient_boosting_model(self):
        X = []
        y = []
        for teacher, subject in zip(self.data['Teacher'], self.data['Subject']):
            for day in self.days:
                for period in range(self.periods):
                    feature = [self.data['Teacher'].index(teacher), self.data['Subject'].index(subject), self.day_mapping[day], period]
                    X.append(feature)
                    y.append(random.choice([0, 1]))  # Random labels for demonstration purposes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        return model, ['Teacher', 'Subject', 'Day', 'Period']

    def train_naive_bayes_model(self):
        X = []
        y = []
        for teacher, subject in zip(self.data['Teacher'], self.data['Subject']):
            for day in self.days:
                for period in range(self.periods):
                    feature = [self.data['Teacher'].index(teacher), self.data['Subject'].index(subject), self.day_mapping[day], period]
                    X.append(feature)
                    y.append(random.choice([0, 1]))  # Random labels for conflicts

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    def load_bert_model(self):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer

    def generate_initial_timetable(self):
        timetable = defaultdict(dict)
        for section in range(1, self.sections + 1):
            for day in self.days:
                for period in range(1, self.periods + 1):
                    teacher = random.choice(self.data['Teacher'])
                    subject = random.choice(self.data['Qualified'][self.data['Teacher'].index(teacher)])
                    timetable[section][(day, period)] = (subject, teacher)
        return timetable

    def optimize_with_levels(self, epochs, models_per_epoch, k):
        best_timetable_level_1 = self.optimize(epochs=1, models_per_epoch=models_per_epoch)
        best_timetable_level_2 = self.reinforcement_learning_optimization(best_timetable_level_1, epochs)
        best_timetable_level_3 = self.recursive_naive_bayes_optimization(best_timetable_level_2, epochs)
        best_timetable_level_4 = self.markov_optimization(best_timetable_level_3, epochs, recursions=5)
        best_timetable_level_5, best_score = self.lda_optimization(best_timetable_level_4, iterations=5)
        best_timetable_level_6, final_score = self.bert_optimization(best_timetable_level_5)

        self.best_timetable = best_timetable_level_6
        self.best_score = final_score

    def recursive_naive_bayes_optimization(self, timetable, depth):
        if depth <= 0:
            return timetable

        improved_timetable = timetable.copy()
        for section in timetable:
            for (day, period), (subject, teacher) in timetable[section].items():
                feature = [
                    self.data['Teacher'].index(teacher),
                    self.data['Subject'].index(subject),
                    self.day_mapping[day],
                    period
                ]
                if self.naive_bayes_model.predict([feature])[0] == 1:  # Conflict found
                    new_teacher = random.choice(self.data['Teacher'])
                    new_subject = random.choice(self.data['Qualified'][self.data['Teacher'].index(new_teacher)])
                    improved_timetable[section][(day, period)] = (new_subject, new_teacher)

        return self.recursive_naive_bayes_optimization(improved_timetable, depth - 1)

    def markov_optimization(self, timetable, epochs, recursions=1):
        improved_timetable = timetable.copy()
        for _ in range(recursions):
            for section in improved_timetable:
                for (day, period), (subject, teacher) in improved_timetable[section].items():
                    if random.random() > 0.5:  # Simulating a decision
                        new_teacher = random.choice(self.data['Teacher'])
                        new_subject = random.choice(self.data['Qualified'][self.data['Teacher'].index(new_teacher)])
                        improved_timetable[section][(day, period)] = (new_subject, new_teacher)
        return improved_timetable

    def lda_optimization(self, timetable, iterations=5):
        best_timetable = timetable
        best_score = self.calculate_score(self.check_conflicts(timetable))

        for _ in range(iterations):
            data_for_lda = []
            for section in best_timetable:
                section_data = []
                for (day, period), (subject, teacher) in best_timetable[section].items():
                    section_data.append(self.data['Subject'].index(subject))
                data_for_lda.append(section_data)

            lda_model = LatentDirichletAllocation(n_components=self.sections, random_state=42)
            lda_model.fit(data_for_lda)

            new_timetable = self.generate_initial_timetable()

            current_score = self.calculate_score(self.check_conflicts(new_timetable))
            if current_score > best_score:
                best_score = current_score
                best_timetable = new_timetable

        return best_timetable, best_score

    def bert_optimization(self, timetable):
        return timetable, self.calculate_score(self.check_conflicts(timetable))

    def optimize(self, epochs, models_per_epoch):
        return self.generate_initial_timetable()

    def reinforcement_learning_optimization(self, timetable, epochs):
        return timetable

    def check_conflicts(self, timetable):
        conflicts = []
        for section in timetable:
            assigned = defaultdict(list)
            for (day, period), (subject, teacher) in timetable[section].items():
                assigned[(day, period)].append((subject, teacher))
            for (day, period), entries in assigned.items():
                if len(entries) > 1:
                    for entry in entries:
                        conflicts.append((section, (day, period), entry))
        return conflicts

    def calculate_score(self, conflicts):
        return 100 - len(conflicts)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/generate_timetable', methods=['POST'])
def generate_timetable():
    data_file = request.files['data_file']
    sections = int(request.form['sections'])
    periods = int(request.form['periods'])
    
    # Load the JSON data from the uploaded file
    data = json.load(data_file)

    # Define the days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Create an instance of the optimizer
    optimizer = TimetableOptimizer(data, sections, days, periods)

    # Optimize the timetable
    optimizer.optimize_with_levels(epochs=10, models_per_epoch=5, k=6)

    # Retrieve the generated timetable, conflicts, and final score
    timetable = optimizer.best_timetable
    conflicts = optimizer.check_conflicts(timetable)
    score = optimizer.best_score

    # Save timetable to a CSV file
    timetable_file_path = os.path.join(DOWNLOAD_FOLDER, 'timetable.csv')
    with open(timetable_file_path, 'w') as f:
        f.write('Section,Day,Period,Subject,Teacher\n')
        for section, schedule in timetable.items():
            for (day, period), (subject, teacher) in schedule.items():
                f.write(f'{section},{day},{period},{subject},{teacher}\n')

    print(f"Timetable saved at: {timetable_file_path}")  # Confirm file creation

    return render_template('form.html', timetable=timetable, conflicts=conflicts, score=score, timetable_file='timetable.csv')

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
