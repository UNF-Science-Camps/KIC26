import random
def get_data_from_file_to_dict(file_path,data_dict={}):
    file = open(file_path, "r",encoding="utf-8")
    for line in file:
        #Doing a lot of splits to get the correct info into the data dict
        current_word,rest=line.split("S1.S")
        amount,rest2=rest.split("S2.S")
        words=rest2.split("S,S")

        #Checking if the key word exsit in the dict
        #One thing to note is that it might be the samme time as when just taking it from the story since it still has to add each of them
        if current_word in data_dict.keys():
            data_dict[current_word][0]+=amount
        else:
            data_dict[current_word]=[amount,{}]
        #It would have to through each word
        for text in words:
            #There might be a new line
            if text =="\n":
                continue
            #Splits it into the word and the amount

            word,amount=text.split("S:S")
            #Again checking if it exsit 
            if word in data_dict[current_word][1].keys():
                data_dict[current_word][1][word]+=amount
            else:
                data_dict[current_word][1][word]=amount
    return data_dict


def chance_fromel(r,total_amount,dif_amount,n):
    if r<=1:
        #This flips it
        r=abs(1-r)
        return n/((total_amount)/(dif_amount*r+(1-r)))
    else:
        r=r-1
        #goes from n to 1
        #n/(n*(n*r+1*(1-r)))
        t=n/(n*(n*r+1*(1-r)))

        #goes from total_amount to 1 as r goes from 1 to 2 in the input
        n=total_amount/(total_amount*(1/(total_amount*r+1*(1-r))))
        #The k aka dif_amount that need to be multiplied to n
        k=(dif_amount*r)+1*(1-r)
        n= n*k
        return t/n
    

#Between 0 and 1 it goes from determinitsk to random by order(0 is full determinsitk)
#Between 1 and 2 it goes from random to full random
def generete_tex_from_dict(data_dict,amount=10,start_word=None,randomnes=1):
    #Making a list to waht the news values would be
    word_list=[]
    if start_word is None:
        start_word =list(data_dict.keys())[0]
    
    prev_word=start_word
    #The loop that actully generetes the different values
    print(start_word)
    while amount>0:
        if prev_word not in data_dict.keys():
            #word_list.append("\n")
            #break
            prev_word=random.choice(list(data_dict.keys()))
        total_amount=data_dict[prev_word][0]
        sorted_dict=dict(sorted(data_dict[prev_word][1].items(),key=lambda item: item[1],reverse=True))
        random_value=random.random()
        #A value to the the sum of the chance
        sum_chance=0
        #looping through the dict 
        #print(sorted_dict)
        for key in sorted_dict.keys():
            sum_chance+=chance_fromel(randomnes,int(total_amount),len(sorted_dict),int(sorted_dict[key]))
            if sum_chance>random_value:
                word_list.append(key)
                prev_word=key
                break
        amount-=1
        #print(data_dict[prev_word])
    print(word_list)
        
import time
tid=time.time()
data=get_data_from_file_to_dict("data_han_solo.txt")

print(time.time()-tid)
#0.0005669593811035156
#0.0004794597625732422

generete_tex_from_dict(data,amount=150,randomnes=1,start_word="rape")