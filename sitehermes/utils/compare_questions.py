import spacy
from utils.GetQuestionsAmazon import run_amazon
from operator import itemgetter

nlp_ptbr = spacy.load("pt_core_news_sm")

def compare_question(question, list_of_questions):
    question = nlp_ptbr(question)
    compare_score_list = []
    for question_template in list_of_questions:
        compare_score_list.append([question.similarity(nlp_ptbr(question_template["pergunta"])), question_template])
    return compare_score_list


def compare_question_amazon_product(question):
    list_of_question = []
    for file in run_amazon():
        for faq in file:
            list_of_question.append(faq)
    best_questions_list = sorted(compare_question(question, list_of_question), key=itemgetter(0))
    print(best_questions_list)
    return best_questions_list[-1]


if __name__ == '__main__':
    compare_question_amazon_product("Funciona para iphone?")