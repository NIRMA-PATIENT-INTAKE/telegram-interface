import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

import os

import joblib
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters

from distool.feature_extraction import SmartSymptomExtractor
from distool.feature_extraction.symptom_collection import SymptomCollection
from distool.feature_extraction.symptom_status import SymptomStatus
from distool.interpretation.explainer import SymptomBasedExplainer
from distool.estimators import DiseaseClassifier

load_dotenv()

# Инициализация классификатора и векторизатора
symptom_vectorizer = SmartSymptomExtractor()
classifier = joblib.load("models/10-06-2023-classifier.joblib")
explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
morph = MorphAnalyzer()
stop_words = set(stopwords.words('russian')) - set(['нет', 'не'])


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('''
    Здравствуйте! 
    Перед вами демонстратор возможностей модуля PATIENT-INTAKE (https://github.com/niRMA-PATIENT-INTAKE/disease/). 
    Он создан для разработчиков медицинских чат-ботов. 
    Это значит, что он умеет выделять симптомы, выносить предварительный диагноз, маркировать срочность приема. 
    часть данных при подключении модуля к реальному проекту должна уходить в информационную систему клиники, 
    а другая часть может быть использована для поддержания естественного диалога с пациентом, 
    обратившимся в диалоговый агент для записи к врачу.
    '''.strip())
    update.message.reply_text('''
    Ниже вы можете ввести строчку с описанием самочувствия на естественном языке и получить ответ от системы.
    Пример строки: «у меня болит голова, сложно фокусироваться, но температуры нет».
    ''')
    context.user_data['symptoms'] = ''


def preprocess_text(text):
    # Токенизация текста и приведение слов к нормальной форме
    messages = text.split("\.")
    print(messages)
    words = ". ".join([
        " ".join([morph.parse(word)[0].normal_form for word in message.split()])
        for message in messages
    ])
    return words


def handle_message(update: Update, context: CallbackContext) -> None:
    # Пользовательский текст
    user_text = update.message.text
    # Предобработка текста
    user_text = preprocess_text(user_text)
    print(user_text)

    # Добавляем новые симптомы к старым
    context.user_data['symptoms'] = context.user_data.get('symptoms', '')
    context.user_data['symptoms'] += '. ' + user_text

    print(context.user_data['symptoms'])

    # Векторизация симптомов
    features = symptom_vectorizer.transform([context.user_data['symptoms']])

    # Предсказание болезни
    predicted_disease = classifier.predict(features)[0]
    confidence = max(classifier.predict_proba(features)[0])
    print(confidence)

    # Если уверенность меньше 60%
    if confidence < 0.60:
        predicted_disease_idx = np.argmax(classifier.predict_proba(features)[0])
        # Получаем веса для этой болезни
        disease_weights = np.abs(classifier.log_reg.coef_[predicted_disease_idx])

        # Получаем индексы трех наиболее важных симптомов
        top_symptoms_indices = disease_weights.argsort()[::-1]

        # Получаем названия этих симптомов
        top_symptoms = [SymptomCollection.get_symptoms()[i].id_name for i in top_symptoms_indices]

        symptom_analysis = list(zip(SymptomCollection.get_symptoms(), features[0]))
        has_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.YES.value
        ]
        no_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.NO.value
        ]
        conf_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.CONFUSED.value
        ]
        question = ''
        if has_symptoms:
            question += f"Я пока определил следующие симптомы: {', '.join(has_symptoms)}"
        if no_symptoms:
            if has_symptoms:
                question += "\nИ "
            question += f"Вы отрицаете следующие симптомы: {', '.join(no_symptoms)}"
        if conf_symptoms:
            question += f"\nЯ пока не уверен про следующие симптомы: {', '.join(conf_symptoms)}"

        to_ask = []

        for sympt in top_symptoms:
            if sympt not in (has_symptoms + no_symptoms + conf_symptoms):
                to_ask.append(sympt)

        question += '\nМне не хватает данных для точного диагноза. Можете ли вы описать свои симптомы более подробно?'
        question += f"\nПодумайте, может у вас наблюдает один из этих сипмтомов: {', '.join(to_ask[:3])}?"

        update.message.reply_text(question.strip())
    else:
        # Объяснение предсказания
        explanation = explainer.explain(features[0])

        # Отправка ответа пользователю
        update.message.reply_text(f"Предварительный диагноз: {predicted_disease}\n{explanation}")
        update.message.reply_text('''
        Напоминаем вам, что вы находитесь в режиме демонстратора работы модуля, 
        созданного для разработчиков медицинских чат-ботов, 
        это означает, что модуль требует дополнительной юстировки
        со стороны клиники, вывод системы не может приниматься за диагноз без консультации с врачом!
        '''.strip())
        context.user_data['symptoms'] = ''


def main() -> None:
    updater = Updater(token=os.environ["TG_BOT_TOKEN"], use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
