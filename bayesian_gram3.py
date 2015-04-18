# coding=UTF-8

import time
import math
from nltk.util import ngrams
from collections import defaultdict
# Read moves from file


def test():
    moves = defaultdict(list)

    moves_data = open('moves_data_initial.txt', 'r')
    while True:
        keyword = moves_data.readline()
        if keyword != '':
            temp_line = keyword.split(' ')
            # temp_list 存BPMRC的count or 機率 0.7 0.1 0.1 0.05 0.05
            temp_list = []
            # print "temp_line=\"" + str(temp_line) + "\" and temp_list= \"" +
            # str(temp_list)

            # 將前面當作4-gram (major current focus in 當作dict的key
            # Does not include 9 when using range() in python
            for i in range(4, 9):
                temp_list.append(float(temp_line[i]))
            # print temp_list
            moves[temp_line[0], temp_line[1],
                  temp_line[2], temp_line[3]] = temp_list

        else:
            break
    # print len(moves)
    moves_data.close()

    # Read paragraph from file
    sent = defaultdict(list)
    input_sentance = []
    f = open('citeseerx_descriptions.txt', 'r')
    article_count = 0

    sentences_count = 0
    while article_count < 200:
        start_time = time.time()
        # 因為第100篇的paragraph被抽出來當testing 所以要跳過training
        if article_count == 100:
            article_count = article_count + 1
        else:
            i = f.readline()
            i = i.lower()
            i = i.replace(',', '')
            i = i.replace('(', '')
            i = i.replace(')', '')
            i = i.replace('\\', '')
            i = i.replace('`', '')
            i = i.replace(';', '')
            i = i.replace("\"", '')
            i = i.replace("[", '')
            i = i.replace("]", '')
            i = i.replace(":", '')
            #i = i.replace(' .','.')
            if i != '':
                input_sentance = i.rstrip()
                input_sentance = i.split('. ')
                # 將句子切成4-gram 放入sent{} 當作value
                n = 4
                for value in input_sentance:
                    fourgram = ngrams(value.split(), n)
                    if article_count == 522:
                        print "value: " + str(value)
                    # 計算paragraph總共有幾句句子，才能從中選出可能為B的句子
                    if len(value) > 10 and len(value.split()) >= 4 and value not in sent:
                        sentences_count = sentences_count + 1
                        for gram in fourgram:
                            # sent.setdefault(value,[]).append(gram)
                            sent[value].append(gram)

            else:
                break
            # 將句子的總數依照b佔句子整體的10% p佔句子整體的30% 依此類推
            max_BPMRC = []
            # B_max, P_mac, M....R...C
            max_BPMRC.append(int(round(sentences_count * 0.1)))
            max_BPMRC.append(int(round(sentences_count * 0.3)))
            max_BPMRC.append(int(round(sentences_count * 0.3)))
            max_BPMRC.append(
                sentences_count - (2 * max_BPMRC[0] + max_BPMRC[1] + max_BPMRC[2]))
            max_BPMRC.append(int(round(sentences_count * 0.1)))

            # B[],P[],M[],R[],C[]
            result_moves = [[], [], [], [], []]

            sentances_in_this_round = []

            # 利用bayesian 計算
            for sentanse in sent:
                BPMRC_TOTAL = [0.15, 0.25, 0.2, 0.25, 0.15]
                # print sent[sentanse]
                gram_len = 1.0 / len(sent[sentanse])
                sentances_in_this_round.append(sentanse)
                for gram in sent[sentanse]:
                    # 若gram存在於moves={...}
                    if gram in moves:
                        gram_probability = float(moves[gram][
                                                 0]) + float(moves[gram][1]) + float(moves[gram][2]) + float(moves[gram][3]) + float(moves[gram][4])
                        for i in range(0, 5):
                            if moves[gram][i] == 0:
                                mini_probability = 0.01 / gram_probability
                                grams_movesprobability_byexp = math.log(
                                    mini_probability, 10)
                                BPMRC_TOTAL[i] = BPMRC_TOTAL[i] + \
                                    grams_movesprobability_byexp - \
                                    math.log(gram_len, 10)
                            else:
                                moves_probability = moves[
                                    gram][i] / gram_probability
                                grams_movesprobability_byexp = math.log(
                                    moves_probability, 10)
                                BPMRC_TOTAL[i] = BPMRC_TOTAL[i] + \
                                    grams_movesprobability_byexp - \
                                    math.log(gram_len, 10)
                    # 若gram不存在moves={...}
                    else:
                        for i in range(0, 5):
                            mini_probability = 0.01
                            grams_movesprobability_byexp = math.log(
                                mini_probability, 10)
                            BPMRC_TOTAL[i] = BPMRC_TOTAL[i] + \
                                grams_movesprobability_byexp - \
                                math.log(gram_len, 10)

                for j in range(0, 5):
                    result_moves[j].append(BPMRC_TOTAL[j])

            #print (sent)

            # 所有句子當中的B，若句子A1的B為最大值，則將A1 tag 為B
            already_found = []
            # print "Moves is B: "
            for move_indicator in range(0, 5):
                BPMRC = ['B', 'P', 'M', 'R', 'C']
                # move_indicator 找出最大值的句子位置
                while max_BPMRC[move_indicator] != 0:
                    update_word_list = []
                    current_max_move_index = result_moves[
                        move_indicator].index(max(result_moves[move_indicator]))
                    # 若是該句已經被tag，則尋找第二高的句子為B
                    if current_max_move_index in already_found:
                        result_moves[move_indicator][
                            current_max_move_index] = -10000
                    else:
                        already_found.append(current_max_move_index)
                        #target_sentences = sent.items()[current_max_move_index][0]
                        target_sentences = sentances_in_this_round[
                            current_max_move_index]
                        # print "target_sentences2: " + target_sentences2
                        # print "target_sentences: " + target_sentences
                        #print [current_max_move_index][0]
                        # print target_sentences
                        sent[target_sentences].append(BPMRC[move_indicator])
                        # print BPMRC[move_indicator]
                        # print sent[target_sentences]
                        m_sent_len = len(sent[target_sentences])

                        # update_word_list=[]
                        # for i in range(0,m_sent_len-2):
                        # update_word_list.append(sent[target_sentences][i])
                        update_word_list = []
                        update_word_list.append(
                            sent[target_sentences][0:m_sent_len - 2])
                        # print "update_word_list2 = " + str(update_word_list2)
                        max_BPMRC[move_indicator] = max_BPMRC[
                            move_indicator] - 1
                        # print "update_word_list= " + str(update_word_list)
                        word_len = len(update_word_list[0])

                        # 將sentences上的4-gram update到moves={...}上
                        for i in range(0, word_len):
                            gram = update_word_list[0][i]

                            # 若gram存在於moves={...}
                            if gram in moves:
                                update_list = moves.get(gram)
                                if move_indicator == 0:
                                    update_list[0] = update_list[0] + 1
                                elif move_indicator == 1:
                                    update_list[1] = update_list[1] + 1
                                elif move_indicator == 2:
                                    update_list[2] = update_list[2] + 1
                                elif move_indicator == 3:
                                    update_list[3] = update_list[3] + 1
                                elif move_indicator == 4:
                                    update_list[4] = update_list[4] + 1
                            elif gram == '':
                                break
                            # 若不存在moves={...}
                            else:
                                if move_indicator == 0:
                                    add_list = {
                                        gram: [1.0, 0.0, 0.0, 0.0, 0.0]}
                                    moves.update(add_list)

                                elif move_indicator == 1:
                                    add_list = {
                                        gram: [0.0, 1.0, 0.0, 0.0, 0.0]}
                                    moves.update(add_list)

                                elif move_indicator == 2:
                                    add_list = {
                                        gram: [0.0, 0.0, 1.0, 0.0, 0.0]}
                                    moves.update(add_list)

                                elif move_indicator == 3:
                                    add_list = {
                                        gram: [0.0, 0.0, 0.0, 1.0, 0.0]}
                                    moves.update(add_list)

                                elif move_indicator == 4:
                                    add_list = {
                                        gram: [0.0, 0.0, 0.0, 0.0, 1.0]}
                                    moves.update(add_list)

            article_count = article_count + 1
            print ("article_count: " + str(article_count))
            print("--- %s seconds ---" % (time.time() - start_time))
    f.close()

    # 計算完10篇文章之後，將moves={...} 寫入file 當作下一次的initial 先驗機率
    fileopen = open('moves_data71.txt', 'w')
    # len(moves)
    print ("moves len: " + str(len(moves)))
    for gram in moves.iterkeys():
        strli = ' '.join(gram)
        new_count = moves.get(gram)
        li = str(new_count[0]) + ' ' + str(new_count[1]) + ' ' + \
            str(new_count[2]) + ' ' + \
            str(new_count[3]) + ' ' + str(new_count[4])
        fileopen.write(strli + ' ' + li + '\n')
    # fileopen.write(str(sent))
    fileopen.close()

    # print moves
start_time = time.time()
test()
print("--- %s Total seconds ---" % (time.time() - start_time))
