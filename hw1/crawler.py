#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import time
import random
#import jsonlines
import json
import sys
import os
import re

#url = "https://www.ptt.cc/bbs/Beauty/index.html"
reg = requests.Session()
payload = {
    'yes': 'yes'
}
reg.post('https://www.ptt.cc/ask/over18', data=payload)
prefix = "https://www.ptt.cc"

def crawl():
    flag = 0
    url = "https://www.ptt.cc/bbs/Beauty/index.html"
    articles_titles = []
    articles_url = []
    articles_hls = []
    articles_dates = []

    while flag == 0 or flag == 1:
        res = reg.get(url)
        content = res.text
        soup = BeautifulSoup(content, "html.parser")
        #articles_title = soup.find_all(class_ = "title")
        next_page = soup.find_all(class_ = "btn wide")[1].get('href')
        url = prefix+next_page
        articles = soup.find_all(class_ = "r-ent")
        #print(articles)
        for article in articles:
            if len(article.find_all(class_ = "title")[0].find_all('a'))!= 0:
                title = article.find_all(class_ = "title")[0].find_all('a')[0].string
                if(title.find('公告')== -1):
                    art_url = article.find_all(class_ = "title")[0].find_all('a')[0].get('href')
                    date = article.find_all(class_ = "date")[0].text
                    nrec = article.find_all(class_ = "nrec")
                    pop = nrec[0].text if nrec[0].text != '' else '0'
                    print(f'{title} {date} {pop}')
                    articles_titles.append(title)
                    articles_dates.append(date)
                    articles_url.append(art_url)
                    articles_hls.append(pop)
                    if date ==  '12/31':
                        flag = 1 if len(articles_titles)< 4000 else -1
        print(next_page)
        d = random.uniform(0.1, 0.2)
        time.sleep(d)
    articles_year = []

    for i,d in enumerate(articles_dates):
        month, day = d.split('/')
        if i < len(articles_dates)/2:
            if  int(month) < 6:
                articles_year.append(2023)
            else:
                articles_year.append(2022)
        else:
            if int(month) < 12:
                articles_year.append(2022)
            else:
                articles_year.append(2021)
    titles_2022 = []
    dates_2022 = []
    url_2022 = []
    hl_2022 = []
    for i in range(len(articles_url)):
        if articles_year[i] == 2022:
            titles_2022.append(articles_titles[i])
            hl_2022.append(articles_hls[i])
            url_2022.append(prefix+articles_url[i])
            month, day = articles_dates[i].split('/')
            tmp = ''
            if month[0] == ' ':
                tmp = '0'+month[1]+ day
            else:
                tmp = month+ day
            dates_2022.append(tmp)
    # with jsonlines.open("all_article.jsonl", mode='w') as w:
    #     # for i in range(len(url_2022)):
    #     #         line = {
    #     #             "date": dates_2022[i],
    #     #             "title": titles_2022[i],
    #     #             "url": url_2022[i],
    #     #         }
    #     #         w.write(line)

    for i in range(len(url_2022)):
        line = {
            "date": dates_2022[i],
            "title": titles_2022[i],
            "url": url_2022[i],
        }
        with open("all_article.jsonl", 'a') as f:
            f.write(json.dumps(line, ensure_ascii=False)+ '\n')

    # with jsonlines.open("all_popular.jsonl", mode='w') as w:
    #     for i in range(len(url_2022)):
    #         if hl_2022[i] == '爆':
    #             line = {
    #                 "date": dates_2022[i],
    #                 "title": titles_2022[i],
    #                 "url": url_2022[i],
    #             }
    #             w.write(line)

    for i in range(len(url_2022)):
        if hl_2022[i] == '爆':
            line = {
                "date": dates_2022[i],
                "title": titles_2022[i],
                "url": url_2022[i],
            }
            with open("all_popular.jsonl", 'a') as f:
                f.write(json.dumps(line, ensure_ascii=False)+ '\n')

def push(start_date,end_date):
    if os.path.isfile("all_article.jsonl")== False:
        crawl()

    url_push = []
    # with json.open('all_article.jsonl') as reader:
    #     for obj in reader:
    #         #print(obj['date']>=start_date and obj['date']<=end_date)
    #         if obj['date']>=start_date and obj['date']<=end_date:
    #             url_push.append(obj['url'])
    with open('all_article.jsonl','r') as f:
        for o in f:
            obj = json.loads(o)
            if obj['date']>=start_date and obj['date']<=end_date:
                url_push.append(obj['url'])
    good = []
    bad = []
    i=0
    for url in url_push:
        print(i)
        i+=1
        res = reg.get(url)
        content = res.text
        soup = BeautifulSoup(content, "html.parser")
        comments = soup.find_all(class_ = "push")
        for comment in comments:
            bads = comment.find_all(class_ = "f1 hl push-tag")
            goods = comment.find_all(class_ = "hl push-tag")
            user = comment.find_all(class_ = "f3 hl push-userid")[0].text
            if len(bads)==1 and bads[0].text == '噓 ':
                bad.append(user)
            elif len(goods)==1 and goods[0].text == '推 ':
                good.append(user)
        d = random.uniform(0.1, 0.3)
        time.sleep(d)
    good_dict = {}
    for g in set(good):
        good_dict[g] = good.count(g)

    bad_dict = {}
    for b in set(bad):
        bad_dict[b] = bad.count(b)
    good_list = sorted(good_dict.items(),key=lambda x: (x[1],x[0][0]),reverse = True)
    bad_list = sorted(bad_dict.items(),key=lambda x: (x[1],x[0][0]),reverse = True)
    push_ans= {
        "all_like": len(good),
        "all_boo": len(bad),
        "like 1":{"user_id": good_list[0][0], "count":good_list[0][1] }, 
        "like 2":{"user_id": good_list[1][0], "count":good_list[1][1] }, 
        "like 3":{"user_id": good_list[2][0], "count":good_list[2][1] }, 
        "like 4":{"user_id": good_list[3][0], "count":good_list[3][1] }, 
        "like 5":{"user_id": good_list[4][0], "count":good_list[4][1] }, 
        "like 6":{"user_id": good_list[5][0], "count":good_list[5][1] }, 
        "like 7":{"user_id": good_list[6][0], "count":good_list[6][1] }, 
        "like 8":{"user_id": good_list[7][0], "count":good_list[7][1] }, 
        "like 9":{"user_id": good_list[8][0], "count":good_list[8][1] }, 
        "like 10":{"user_id": good_list[9][0], "count":good_list[9][1] },
        "boo 1":{"user_id": bad_list[0][0], "count":bad_list[0][1] },
        "boo 2":{"user_id": bad_list[1][0], "count":bad_list[1][1] },
        "boo 3":{"user_id": bad_list[2][0], "count":bad_list[2][1] },
        "boo 4":{"user_id": bad_list[3][0], "count":bad_list[3][1] },
        "boo 5":{"user_id": bad_list[4][0], "count":bad_list[4][1] },
        "boo 6":{"user_id": bad_list[5][0], "count":bad_list[5][1] },
        "boo 7":{"user_id": bad_list[6][0], "count":bad_list[6][1] },
        "boo 8":{"user_id": bad_list[7][0], "count":bad_list[7][1] },
        "boo 9":{"user_id": bad_list[8][0], "count":bad_list[8][1] },
        "boo 10":{"user_id": bad_list[9][0], "count":bad_list[9][1] },
    }

    json_object = json.dumps(push_ans, indent= 4)
    with open(f"push_{start_date}_{end_date}.json", "w") as outfile:
        outfile.write(json_object)

def popular(start_date,end_date):
    if os.path.isfile("all_article.jsonl")== False:
        crawl()
    url_pop = []
    with open('all_popular.jsonl','r') as f:
        for o in f:
            obj = json.loads(o)
            if obj['date']>=start_date and obj['date']<=end_date:
                url_pop.append(obj['url'])
    image_url = []
    for url in url_pop:
        print(url)
        res = reg.get(url)
        content = res.text
        soup = BeautifulSoup(content, "html.parser")
        images = soup.find_all("a")
        for image in images:
            if image.text.find('https://') != -1 or image.text.find('http://') != -1:
                if image.text.endswith('.jpeg') or image.text.endswith('.jpg') or image.text.endswith('.png') or image.text.endswith('.gif'):
                    image_url.append(image.text)
        d = random.uniform(0.1, 0.4)
        time.sleep(d)
    pop_dict = {
        "number_of_popular_articles":len(url_pop),
        "image_urls": image_url,
    }

    json_object = json.dumps(pop_dict, indent=4)

    with open(f"popular_{start_date}_{end_date}.json", "w") as outfile:
        outfile.write(json_object)

def keyword(start_date,end_date,key_word):
    url_key = []
    # with json.open('all_article.jsonl') as reader:
    #     for obj in reader:
    #         #print(obj['date']>=start_date and obj['date']<=end_date)
    #         if obj['date']>=start_date and obj['date']<=end_date:
    #             url_key.append(obj['url'])
    with open('all_article.jsonl','r') as f:
        for o in f:
            obj = json.loads(o)
            if obj['date']>=start_date and obj['date']<=end_date:
                url_key.append(obj['url'])
    key_ans=[]
    for url in url_key:
        #print(url)
        res = reg.get(url)
        content = res.text
        soup = BeautifulSoup(content, "html.parser")
        content = soup.find_all(class_ = "bbs-screen bbs-content")
        if key_word in content[0].text.split('發信站')[0]:
            print(url)
            main_content = content[0].text
            pattern = '(https?:\/\/.*?\.(?:png|jpg|gif|jepg))'
            imgs = re.findall(pattern, main_content)
            for img in imgs:
                #print(img)
                key_ans.append(img)
        else:
            d = random.uniform(0.2, 0.4)
            time.sleep(d)
    key_dict = {
        "image_urls": key_ans,
    }

    json_object = json.dumps(key_dict, indent=4)

    with open(f"keyword_{key_word}_{start_date}_{end_date}.json", "w") as outfile:
        outfile.write(json_object)




if __name__ == '__main__':
    if sys.argv[1] == 'crawl':
        crawl()
    elif sys.argv[1] == 'push':
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        push(start_date,end_date)
    elif sys.argv[1] == 'popular':
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        popular(start_date,end_date)
    elif sys.argv[1] == 'keyword':
        key_word = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]
        keyword(start_date,end_date,key_word)
    else:
        print('Error!')