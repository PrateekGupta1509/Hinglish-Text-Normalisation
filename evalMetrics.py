#-*- coding: utf-8 -*-
#!/usr/bin/env python


# Script adapted from here https://github.com/zszyellow/WER-in-python
import sys
import numpy
from nltk.translate import bleu_score
import chrf_score


def editDistance(r, h):
    '''
    This funciton is to calculate the edit distance of refernce sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1] and x >= 1 and y >= 1: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif d[x][y] == d[x][y-1]+1 and y >= 1:
            list.append("i")
            x = x
            y = y - 1
        elif d[x][y] == d[x-1][y-1]+1 and x >= 1 and y >= 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result

def chrf(r, h, ret_all = False):
    try:
        score, precision, recall = chrf_score.sentence_chrf(r,h)
        if ret_all:
            return score, precision, recall
        return score
    except:
        return 0.0

def bleu(r, h):
    try:
        return bleu_score.sentence_bleu(r,h)
    except:
        return 0.0