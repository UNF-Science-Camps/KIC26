import time
import re
tid=time.time()

#Function to lower all values
def lower_values(list):
    new_list=[]
    for value in list:
        new_list.append(value.lower())
    return new_list


def generate_dict(list_of_words,data_dict={},amount=1):
    """
    The dict would have a list of tuples which each have a word and a number, which represtinen  the amount
    """
    for index,word_arr in  enumerate(list_of_words):
        print(word_arr)
        word=" ".join(word_arr)
        #making sure that it does snot goes out of bound becomes of something later, yes this does remove some info but it's so small it does not matter
        if index==len(list_of_words)-1:
            break
        if word in data_dict:
            #Word is a key 1 is the number where the dict is and the 0 is the total amount
            data_dict[word][0]+=1
            #Finding the word and adding one to the amount if it's a key
            if list_of_words[index+1] in data_dict[word][1]:
                data_dict[word][1][list_of_words[index+1]]+=1
            else:
                data_dict[word][1][list_of_words[index+1]]=1
        else:
            data_dict[word]=[1,{}]
            data_dict[word][1][list_of_words[index+1]]=1
    return data_dict


def save_dict_to_file(data_dict):
    file = open('data_han_solo.txt', 'w',encoding="utf-8")
    #Looping through each dict and writing it in the file in a way that can be read again
    for key1 in data_dict:
        #Writing the OG word
        file.write(key1)
        #Writing a sepereater all words are lowerd so S1 would never be there
        file.write("S1.S")
        #Writing the amount of total then a sepereter
        file.write(str(data_dict[key1][0]))
        file.write("S2.S")
        #Now to write each of the subkeys
        for subkey in data_dict[key1][1]:
            #The sub key 
            file.write(subkey)
            #The number spereter
            file.write("S:S")
            #The number
            file.write(str(data_dict[key1][1][subkey]))
            #The spereter between different values in the dict
            file.write("S,S")
            


        #Making a new line
        file.write("\n")
    file.close()

def make_sub_words(data,amount):
    list_words=[]
    for i in range(len(data)+1-amount):
        #Making a for loop to add the different order set in the word
        temp_ar=[]
        for k in range(amount):
            temp_ar.append(data[i+k])
        list_words.append(temp_ar)
    return list_words


def generete_data_file(file_path):
    #file_path="hansoloshit.txt"
    total_story=""


    file = open(file_path, "r",encoding="utf-8")
    for line in file:
        total_story=total_story+line.strip()
    file.close()
    
    story_list=total_story.split(" ")
    total_story=total_story.lower()
    temp_list=re.split(r"(?![0-9][,\.][0-9])[,\.;\" ]",total_story)
    story_list=[]
    for t in temp_list:
        if t:
            story_list.append(t)
    #print(story_list)
    story_list=make_sub_words(story_list,2)
    
    d=generate_dict(story_list)
    #print(d)
    """
    
    print(time.time()-tid)
    save_dict_to_file(d)
    """

generete_data_file("story.txt")