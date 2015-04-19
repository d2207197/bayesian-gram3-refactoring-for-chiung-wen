# coding=UTF-8

from __future__ import division
import time
import math
from nltk.util import ngrams as gen_ngrams
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
            moves[
                temp_line[0],
                temp_line[1],
                temp_line[2],
                temp_line[3]] = temp_list

        else:
            break
    # print len(moves)
    moves_data.close()
    return moves


def sent_tokenizer(paragraph):
    paragraph = paragraph.strip().lower().translate(None, ',()\\`;"[]:')

    sentences = paragraph.split('. ')
    return sentences


def get_max_BPMRC(sent_length):
    # 將句子的總數依照 b佔句子整體的10% p佔句子整體的30% 依此類推
    max_BPMRC = [  # B_max, P_mac, M....R...C
        int(round(sent_length * 0.1)),
        int(round(sent_length * 0.3)),
        int(round(sent_length * 0.3))
    ]
    max_BPMRC += [len(sent) - (2 * max_BPMRC[0] + max_BPMRC[1] + max_BPMRC[2]),
                  int(round(sent_length * 0.1))]
    return max_BPMRC


# 利用bayesian 計算
def bayesian(moves_prob, gram_len):
    grams_movesprobability_byexp = math.log10(moves_prob)
    return grams_movesprobability_byexp - gram_len


from heapq import heappush, heappop


def gen_new_moves(sent, moves):
    result_moves = [[], [], [], [], []]
    for sent_idx, (sentence, ngrams, _) in enumerate(sent):
        BPMRC_TOTAL = [0.15, 0.25, 0.2, 0.25, 0.15]
        # print sent[sentence]
        gram_len = math.log10(1 / len(ngrams))
        for gram in ngrams:

            # 若gram存在於moves={...}
            if gram in moves:
                gram_probability = sum(moves[gram])
                for i in range(0, 5):
                    if moves[gram][i] == 0:
                        BPMRC_TOTAL[i] += bayesian(0.01 / gram_probability,
                                                   gram_len)
                    else:
                        BPMRC_TOTAL[i] += bayesian(moves[gram][i] /
                                                   gram_probability, gram_len)

            # 若gram不存在moves={...}
            else:
                for i in range(0, 5):
                    BPMRC_TOTAL[i] += bayesian(0.01, gram_len)

        for j in range(0, 5):
            heappush(result_moves[j], (-BPMRC_TOTAL[j], sent_idx))

            # result_moves[j].append((sent_idx, BPMRC_TOTAL[j]))

    return result_moves


def dual_hash(obj):
    hash_value = hash(obj)
    return hash_value & 0xffffff, hash_value & 0xffffff000000 >> 4 * 6


def in_2lvl_set(two_lvl_set, obj):
    hash1, hash2 = dual_hash(obj)
    return hash1 in two_lvl_set and hash2 in two_lvl_set[hash1]


def add_to_2lvl_set(two_lvl_set, obj):
    hash1, hash2 = dual_hash(obj)
    two_lvl_set[hash1].add(hash2)


from collections import namedtuple

SentData = namedtuple('SentData', ['sentence', 'ngrams', 'moves'])


def moves_update(moves, move_indicator, max_move_sent_ngrams):
    for gram in max_move_sent_ngrams:
        # gram = update_word_list[0][i]

        # 若gram存在於moves={...}
        if gram in moves:
            moves[gram][move_indicator] += 1

        elif gram == '':
            break
        # 若不存在moves={...}
        else:
            moves_gram_data = [0.0] * 5
            moves_gram_data[move_indicator] = 1.0
            moves[gram] = moves_gram_data


if __name__ == '__main__':

    moves = get_moves_data('moves_data_initial.txt')

    # Read paragraph from file
    sent = []
    sent_set = defaultdict(set)

    # while paragraph_count < 200:
    for paragraph_count, paragraph in enumerate(
            open('citeseerx_descriptions.txt.300', 'r')):
        start_time = time.time()

        # 因為第100篇的paragraph被抽出來當testing 所以要跳過training
        if paragraph_count == 100:
            continue

        for sentence in sent_tokenizer(paragraph):
            if len(sentence) > 10 and len(
                    sentence.split()) >= 4 and not in_2lvl_set(sent_set, sentence):

                add_to_2lvl_set(sent_set, sentence)
                sent.append(SentData(
                    sentence, list(gen_ngrams(sentence.split(), 4)), []))

        max_BPMRC = get_max_BPMRC(len(sent))

        # B[],P[],M[],R[],C[]
        result_moves = gen_new_moves(sent, moves)
        # print (sent)

        # 所有句子當中的B，若句子A1的B為最大值，則將A1 tag 為B
        already_found = set()
        # print "Moves is B: "

        for move_indicator in range(0, 5):
            BPMRC = 'BPMRC'
            # move_indicator 找出最大值的句子位置
            while max_BPMRC[move_indicator] != 0:
                # current_max_move_index = max(
                # enumerate(result_moves[move_indicator]),
                # key=itemgetter(1))[0]
                current_max_move_index = heappop(
                    result_moves[move_indicator])[1]

                # current_max_move_index = result_moves[
                #     move_indicator].index(max(result_moves[move_indicator]))
                # 若是該句已經被tag，則尋找第二高的句子為B

                if current_max_move_index in already_found:
                    continue
                    # result_moves[move_indicator][current_max_move_index] = -10000

                already_found.add(current_max_move_index)
                #target_sentences = sent.items()[current_max_move_index][0]
                # target_sentences = sentences_in_this_round[
                # current_max_move_index]
                # print "target_sentences2: " + target_sentences2
                # print "target_sentences: " + target_sentences
                #print [current_max_move_index][0]
                # print target_sentences
                sent[current_max_move_index].moves.append(
                    BPMRC[move_indicator])
                # print BPMRC[move_indicator]
                # print sent[target_sentences]
                # m_sent_len = len(sent[current_max_move_index].ngrams)

                # update_word_list=[]
                # for i in range(0,m_sent_len-2):
                # update_word_list.append(sent[target_sentences][i])
                # update_word_list = []
                # update_word_list.append(
                # sent[current_max_move_index].ngrams)
                # print "update_word_list2 = " + str(update_word_list2)
                max_BPMRC[move_indicator] -= 1
                # print "update_word_list= " + str(update_word_list)
                # word_len = len(update_word_list[0])

                # 將sentences上的4-gram update到moves={...}上
                moves_update(moves, move_indicator,
                             sent[current_max_move_index].ngrams)

        print("paragraph_count: " + str(paragraph_count))
        print("--- %s seconds ---" % (time.time() - start_time))

# 計算完10篇文章之後，將moves={...} 寫入file 當作下一次的initial 先驗機率
    fileopen = open('moves_data71.txt', 'w')
    # len(moves)
    print("moves len: " + str(len(moves)))
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
