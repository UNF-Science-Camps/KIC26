import time
tid=time.time()

#Function to lower all values
def lower_values(list):
    new_list=[]
    for value in list:
        new_list.append(value.lower())
    return new_list


def generate_dict(list_of_words,data_dict={}):
    """
    The dict would have a list of tuples which each have a word and a number, which represtinen  the amount
    """
    for index,word in  enumerate(list_of_words):
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



def generete_data_file(file_path):
    #file_path="hansoloshit.txt"
    total_story=""


    file = open(file_path, "r",encoding="utf-8")
    for line in file:
        total_story=total_story+line.strip()
    file.close()
    story_list=total_story.split(" ")
    story_list=lower_values(story_list)
    d=generate_dict(story_list)
    print(time.time()-tid)
    save_dict_to_file(d)

generete_data_file("Mergedshit.txt")