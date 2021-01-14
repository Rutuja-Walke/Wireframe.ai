import os
output=""
output_file_name = "resources/my_words.vocab"
input_path="./all_data/"
my_dic={}
iter=0
for f in os.listdir(input_path):
    if f.find(".png") == -1:
        file = open(input_path+f, 'r')
        text = file.read()
        text=text.replace(" ","\t")
        text=text.replace("\n","\t")
        text=text.split(sep="\t")
        text.remove('')
        for i in text:
            if(i[-1]!=','):
                my_dic[i]=1
            else:
                my_dic[i[:-1]]=1
        #print(text)
        # iter+=1
        # if iter==2:
        #     break
        file.close()
print(my_dic)
output=", <START> <END> "
for key in my_dic.keys():
    output+=key
    output+=" "
output_file = open(output_file_name, 'w')
output_file.write(output)
output_file.close()

print(my_dic,len(my_dic))