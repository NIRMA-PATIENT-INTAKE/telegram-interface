import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

import os

import joblib
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler,  MessageHandler, CallbackContext, Filters

from distool.feature_extraction import SmartSymptomExtractor
from distool.feature_extraction.symptom_collection import SymptomCollection
from distool.feature_extraction.symptom_status import SymptomStatus
from distool.interpretation.explainer import SymptomBasedExplainer
from distool.estimators import DiseaseClassifier, UrgencyClassifier

load_dotenv()

# Инициализация классификатора и векторизатора
symptom_vectorizer = SmartSymptomExtractor()
disease_classifier = DiseaseClassifier.load("models/disease-classifier.joblib")
urgent_classifier = UrgencyClassifier.load("models/urgent-classifier.joblib")
explainer = SymptomBasedExplainer(symptom_vectorizer, disease_classifier)
morph = MorphAnalyzer()
stop_words = set(stopwords.words('russian')) - set(['нет', 'не'])


WELCOME_TEXT = '''
Здравствуйте! 
Перед вами демонстратор возможностей модуля PATIENT-INTAKE (https://github.com/niRMA-PATIENT-INTAKE/disease/). 
Он создан для разработчиков медицинских чат-ботов. 
Это значит, что он умеет выделять симптомы, выносить предварительный диагноз, маркировать срочность приема. 
часть данных при подключении модуля к реальному проекту должна уходить в информационную систему клиники, 
а другая часть может быть использована для поддержания естественного диалога с пациентом, 
обратившимся в диалоговый агент для записи к врачу.
'''.strip()

HINT_TEXT = '''
Ниже вы можете ввести строчку с описанием самочувствия на естественном языке и получить ответ от системы.
Пример строки: «у меня болит голова, сложно фокусироваться, но температуры нет».
'''.strip()



def start(update: Update, context: CallbackContext, chat_id = None) -> None:
    keyboard = [
        [InlineKeyboardButton("Stop", callback_data='stop_action')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if chat_id:
        context.bot.send_message(
            chat_id=chat_id,
            text=WELCOME_TEXT
        )
        context.bot.send_message(
            chat_id=chat_id,
            text=HINT_TEXT,
            reply_markup=reply_markup
        )
    else:
        update.message.reply_text(WELCOME_TEXT)
        update.message.reply_text(HINT_TEXT, reply_markup=reply_markup)

    context.user_data['symptoms'] = ''
    context.user_data['attempt'] = 0


def stop(update: Update, context: CallbackContext, chat_id = None) -> None:
    keyboard = [
        [InlineKeyboardButton("Start", callback_data='start_action')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if chat_id:
        context.bot.send_message(
            chat_id=chat_id,
            text='''Диалог закончен, выберите действие.''',
            reply_markup=reply_markup
        )
    else:
        update.message.reply_text(
            '''Диалог закончен, выберите действие.''',
            reply_markup=reply_markup
        )
    context.user_data['symptoms'] = ''
    context.user_data['attempt'] = 0


def button(update, context):
    query = update.callback_query

    if query.data == "start_action":
        query.edit_message_text(text="Начали!")
        start(update, context, query.message.chat_id)
    elif query.data == "stop_action":
        # Здесь код для обработки действия stop
        query.edit_message_text(text="Остановлено!")
        stop(update, context, query.message.chat_id)


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
    keyboard = [
        [InlineKeyboardButton("Stop", callback_data='stop_action')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Пользовательский текст
    user_text = update.message.text
    # Предобработка текста
    user_text = preprocess_text(user_text)
    print(user_text)

    # Добавляем новые симптомы к старым
    context.user_data['symptoms'] = context.user_data.get('symptoms', '')
    context.user_data['symptoms'] += '. ' + user_text
    context.user_data['attempt'] = context.user_data.get('attempt', 0) + 1

    print(context.user_data['symptoms'])

    # Векторизация симптомов
    features = symptom_vectorizer.transform([context.user_data['symptoms']])

    # Предсказание болезни
    predicted_disease = disease_classifier.predict(features)[0]
    confidence = max(disease_classifier.predict_proba(features)[0])
    print(confidence)

    if confidence < 0.60:
        predicted_disease_idx = np.argmax(disease_classifier.predict_proba(features)[0])
        # Получаем веса для этой болезни
        disease_weights = np.abs(disease_classifier.log_reg.coef_[predicted_disease_idx])

        # Получаем индексы трех наиболее важных симптомов
        top_symptoms_indices = disease_weights.argsort()[::-1]

        # Получаем названия этих симптомов
        top_symptoms = [SymptomCollection.get_symptoms()[i].id_name for i in top_symptoms_indices]

        symptom_analysis = list(zip(SymptomCollection.get_symptoms(), features[0]))
        has_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.YES.value and s.id_name != "инсульт"
        ]
        no_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.NO.value and s.id_name != "инсульт"
        ]
        conf_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.CONFUSED.value and s.id_name != "инсульт"
        ]

        if context.user_data.get('attempt', 0) > 2:
            if not has_symptoms and not no_symptoms and not conf_symptoms:
                update.message.reply_text("Присутствующая у пациента болезнь в текущей версии модуля не может быть "
                                          "диагностирована, необходимо пополнить обучающие данные на стороне клиники")
                stop(update, context)
                return

            answer = (
                "Cимптомы подходят для нескольких категорий заболеваний, "
                "предварительную диагностику может произвести только специалист."
            )

            if has_symptoms:
                answer += f"\nПрисутствуют симптомы: {', '.join(has_symptoms)}."

            if no_symptoms:
                answer += f"\nОтрицаются следующие: {', '.join(no_symptoms)}."

            if conf_symptoms:
                answer += f"\nПока не понятно: {', '.join(conf_symptoms)}."

            update.message.reply_text(answer)
            stop(update, context)
            return

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

        update.message.reply_text(question.strip(), reply_markup=reply_markup)
    elif predicted_disease in ["орви", "covid-19"]:
        update.message.reply_text("Выявлена группа респираторных заболеваний, есть риск COVID-19,"
                                  " невозможна диагностика с использованием ml-модели")
    else:
        # Объяснение предсказания
        explanation = explainer.explain(features[0])

        # Отправка ответа пользователю
        update.message.reply_text(f"Предварительный диагноз: {predicted_disease}\n{explanation}")

        if predicted_disease in ["инфаркт", "аритмия"]:
            urgent_message = "Мы выявили группу кардиологических заболеваний. Ваш предварительный статус: "
            is_urgent = urgent_classifier.predict(features)[0]

            if is_urgent:
                urgent_message += "срочный. Советуем срочно обратиться к специалисту."
            else:
                urgent_message += "плановый. Это не значит, что не нужно идти к врачу. Совутем незамедлительно " \
                                  "обратиться к специалисту при любом подозрении на ухудшение."
            update.message.reply_text(urgent_message)

        update.message.reply_text('''
        Напоминаем вам, что вы находитесь в режиме демонстратора работы модуля, 
        созданного для разработчиков медицинских чат-ботов, 
        это означает, что модуль требует дополнительной юстировки
        со стороны клиники, вывод системы не может приниматься за диагноз без консультации с врачом!
        '''.strip())
        stop(update, context)


def main() -> None:
    updater = Updater(token=os.environ["TG_BOT_TOKEN"], use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(CallbackQueryHandler(button))

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
