# coding=UTF-8

import time
import math
from nltk.util import ngrams
from collections import defaultdict
# Read moves from file


def get_moves_data(filename):
    moves = defaultdict(list)
    moves_data = open(filename, 'r')
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
    return moves


def sent_tokenizer(paragraph):
    paragraph = paragraph.strip().lower().translate(None, ',()\\`;"[]:')

    sentences = paragraph.split('. ')
    return sentences


def sent_fourgrams(sentence):
    # 將句子切成4-gram 放入sent{} 當作value
    n = 4
    yield sentence, list(ngrams(sentence.split(), n))
    # sent[sentence] = list(fourgrams)


def get_max_BPMRC(sent):
    # 將句子的總數依照 b佔句子整體的10% p佔句子整體的30% 依此類推
    max_BPMRC = [
        # B_max, P_mac, M....R...C
        int(round(len(sent) * 0.1)),
        int(round(len(sent) * 0.3)),
        int(round(len(sent) * 0.3))]
    max_BPMRC += [
        len(sent) - (2 * max_BPMRC[0] + max_BPMRC[1] + max_BPMRC[2]),
        int(round(len(sent) * 0.1))]
    return max_BPMRC


# 利用bayesian 計算
def bayesian(sent):
    result_moves = [[], [], [], [], []]
    for sentence in sent:
        BPMRC_TOTAL = [0.15, 0.25, 0.2, 0.25, 0.15]
        # print sent[sentence]
        gram_len = 1.0 / len(sent[sentence])
        for gram in sent[sentence]:
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

    return result_moves


if __name__ == '__main__':

    moves = get_moves_data('moves_data_initial.txt')

    # Read paragraph from file
    sent = defaultdict(list)

    # while paragraph_count < 200:
    for paragraph_count, paragraph in enumerate(
            open('citeseerx_descriptions.txt.100', 'r')):
        start_time = time.time()

        # 因為第100篇的paragraph被抽出來當testing 所以要跳過training
        if paragraph_count == 100:
            continue

        for sentence in sent_tokenizer(paragraph):
            if len(sentence) > 10 and len(sentence.split()) >= 4 and sentence not in sent:
                sent.update(sent_fourgrams(sentence))

        max_BPMRC = get_max_BPMRC(sent)

        # B[],P[],M[],R[],C[]

        result_moves = bayesian(sent)
        #print (sent)
        sentences_in_this_round = sent.keys()

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
                    target_sentences = sentences_in_this_round[
                        current_max_move_index]
                    # print "target_sentences2: " + target_sentences2
                    # print "target_sentences: " + target_sentences
                    #print [current_max_move_index][0]
                    # print target_sentences
                    sent[target_sentences].append(BPMRC[move_indicator])  # Sent append
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

        print ("paragraph_count: " + str(paragraph_count))
        print("--- %s seconds ---" % (time.time() - start_time))

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
    print("--- %s Total seconds ---" % (time.time() - start_time))
