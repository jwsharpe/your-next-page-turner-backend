import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rake_nltk import Rake
import string
from signin_config import *


# Scraping functions

def get_urls():
    '''gets all the urls from GoodReads top ~200 books from years 1909 to 2019'''
    url = []
    for i in range(1951,2019):
        url.append(f'https://www.goodreads.com/book/popular_by_date/{i}/')
    return url


def initial_gr_signin(driver):
    ''' Signs into goodreads with non-existant account to avoid ensuing pop-ups '''
    driver.get('https://www.goodreads.com/')

    email = driver.find_element_by_xpath('//*[@id="userSignInFormEmail"]')
    email.click()
    email.send_keys(username)

    passwrd = driver.find_element_by_xpath('//*[@id="user_password"]')
    passwrd.click()
    passwrd.send_keys(pswd)

    signin = driver.find_element_by_xpath('//*[@id="sign_in"]/div[3]/input[1]')
    signin.click()
    


def goodreads_list_scrape(driver):
    ''' 
    Takes in a driver pointed to a top ~200 page, and returns initial info (title, authors, num_ratings and id) on 
    all books on the page in the form of a list of dictionaries. Prints progress updates every 50 books.
    '''
    list_of_book_dicts = []
    
    counter = 0
    
    titles_blocks = driver.find_elements_by_class_name('bookTitle')
    
    authors_blocks = driver.find_elements_by_class_name('authorName')
  
    ratings_blocks = driver.find_elements_by_class_name('minirating')
    
    num_blocks = driver.find_elements_by_class_name('minirating')
    
    for i in range(len(titles_blocks)):
        gs_dict = {}
        
        gs_dict['titles'] = titles_blocks[i].text
        
        gs_dict['authors'] = authors_blocks[i].text
            
        if 'really liked it' in ratings_blocks[i].text:
            gs_dict['ratings'] = ratings_blocks[i].text[16:21]
        else:
            gs_dict['ratings'] = ratings_blocks[i].text[0:4]
            
    
        if 'really liked it' in num_blocks[i].text:
            gs_dict['num_ratings'] = num_blocks[i].text[34:]
        else:
            gs_dict['num_ratings'] = num_blocks[i].text[18:]
        
        gs_dict['id'] = re.search(r'\d+',titles_blocks[i].get_attribute('href')).group()
        
        counter += 1
        if counter%50 == 0:
            print(counter)
            
        list_of_book_dicts.append(gs_dict)
        
    return list_of_book_dicts



def secondary_scrape(gr_id, driver):
    '''
    Scrapes the goodreads website with given dic.id to get img, description, format, pages and genre. 
    Funciton returns the variables in that order
    ! not yet tested as a function !
    '''
    img, descrip, form, page, gen = '','','','',''
    
    site = f"https://www.goodreads.com/book/show/{gr_id}"
    driver.get(site)
    
    try:
        img = driver.find_element_by_id('coverImage').get_attribute('src')
        
    except:
        img = ''

    
    # full description - cleaned from newline breaks
    more = driver.find_elements_by_xpath('//*[@id="description"]/a')
    try:
        # handling errors to do with an extended description
        more[0].click()
        try:
            describe = driver.find_element_by_id('description').text[:-7].strip()
            descrip = ' '.join(describe.split('\n'))
        except:
            descrip = ''

    except:
        try:
            describe = driver.find_element_by_id('description').text[:-7].strip()
            descrip = ' '.join(describe.split('\n'))
        except:
            descrip = ''

    
    # format and number of pages
    try:
        details = driver.find_element_by_id('details')
        span = details.find_elements_by_css_selector('span')

        for word in span:
            if len(word.text) >0:
                if 'cover'in word.text:
                    form = word.text
                if 'pages'in word.text:
                    page = word.text
    except:
        form = ''
        page = ''

    # get a list of genres
    try:
        genre_blocks = driver.find_elements_by_class_name('actionLinkLite.bookPageGenreLink')
        genre = []
        for genre_block in genre_blocks:
            if genre_block.text[0].isdigit() == False:
                genre.append(genre_block.text)
        genre = set(genre)
    except:
        genre = ''
        
    return img, descrip, form, page, genre


def get_imgs(driver):
    '''
    Takes in a driver and returns the source image (coverImage) for the page that the driver is currently pointed at. 
    Returns a NaN for instances where the information is not found
    '''
    try:
        img = driver.find_element_by_id('coverImage')
        img = img.get_attribute('src')
        return img
    except:
        try:
            img = driver.find_element_by_class('mainBookCover')
            img = img.get_attribute('src')
            return img
        except:
            try:
                img = driver.find_element_by_xpath('/html/body/div[2]/div[3]/div[1]/div[2]/div[2]/div[1]/div[1]/div[1]/div[1]/a/img')
                img = img.get_attribute('src')
                return img
            except:
                return np.nan

def get_description(driver):
    '''
    Takes in a driver and returns the book description for the page that the driver is currently pointed at. 
    Returns a NaN for instances where the information is not found
    '''
    more = driver.find_elements_by_xpath('//*[@id="description"]/a')
    try:
        # handling errors to do with an extended description
        more[0].click()
        try:
            describe = driver.find_element_by_id('description').text[:-7].strip()
            descrip = ' '.join(describe.split('\n'))
        except:
            descrip = np.nan

    except:
        # if there is no "more" button
        try:
            describe = driver.find_element_by_id('description').text[:-7].strip()
            descrip = ' '.join(describe.split('\n'))
        except:
            descrip = np.nan
    return descrip            
            


def get_genre(driver):
    genre = np.nan
    try:
        genre_blocks = driver.find_elements_by_class_name('actionLinkLite.bookPageGenreLink')
        genre = []
        for genre_block in genre_blocks:
            if genre_block.text[0].isdigit() == False:
                genre.append(genre_block.text)
        genre = ', '.join(genre)
    except:
        pass
    
    return genre

    
def get_form_page_isbn(driver):
    ''' Returns book format, number of pages and isbn number, in that order. '''
    form = np.nan
    page = np.nan
    isbn = np.nan

    try:
        details = driver.find_element_by_id('details')
        span = details.find_elements_by_css_selector('span')

        for word in span:
            if 'hardcover' in word.text.lower() or 'paperback' in word.text.lower():
                form = word.text.lower()
            if 'pages' in word.text.lower():
                page = int(word.text.split()[0])
            if 'ISBN' in word.text:
                isbn = word.text
                isbn = int(isbn.split()[1][:-1])
    except:
        pass

    return form, page, isbn


# Cleaning Functions (helpers to more complex rec system)

def clean_row(row):
    '''cleans the description of punctuation and digits '''
    
    # Cleaning: get rid of punctuations in descriptions
    row.description = row.description.replace('”','')
    row.description = row.description.replace('“','')
    for c in string.punctuation:
        row.description = row.description.replace(c,"")

    # Cleaning: get rid of digits in descriptions
    for s in string.digits:
        row.description = row.description.replace(s,"")

    return row['description']


def make_keywords(row):
    '''Makes keywords of description using "Rake" '''
    plot = row['description']
    # instantiating Rake, by default is uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary with key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    
    # return the key words
    return list(key_words_dict_scores.keys())


def make_BoD(row):
    ''' Makes a bag of words (Bag_of_Description) containing all the contents of the row as a string'''
    words = ''
    colums = row.keys().tolist()
    for col in colums:
        words = words + str(row[col]) + ' '
    return words


def clean_create_BoD(row):
    '''
    Combines helper functions to clean description, make keywords and make a final 
    bag of words (Bag_of_Description) for the row.
    '''
    # make a deep copy so we can manipulate and not recieve warning
    our_row = row.copy(deep=True)
    
    # clean description
    our_row['description'] = clean_row(row)

    # assigning the key words to the new column
    our_row['Key_words'] = make_keywords(our_row)
    
    # make bag of description
    our_row['bag_of_description'] = make_BoD(our_row) 
    
    return our_row
    





# Most simple rec system

def simple_rec(genre, length, popularity, df):
    ''' 
    use a  genre_id_dict and df_simple
    '''
    #make genres into dictionary with titles as index
    genre_dict = dict(df.genre)
    
    # put sub-genres into a main genres dictionary
    genre_id_dict = {'Scifi': [], 'Romance':[], 'Thriller':[], 'Comics':[], 'Biography':[], 'Nonfiction':[], 'Self_help':[], 'Young_Adult':[], 'Family':[], 'Fiction':[]}

    for k,v in genre_dict.items():  # where k=book_name and v=genre_category
        if ('Fantasy' in v) or ('Science' in v):
            genre_id_dict['Scifi'].append(k)
        if ('Romance' in v) or ('Chick Lit' in v) or ('Erotic' in v) or ('Contemporary' in v) or ('Drama' in v):
            genre_id_dict['Romance'].append(k)
        if ('Thriller' in v) or ('Mystery' in v) or ('Crime' in v) or ('Horror' in v):
            genre_id_dict['Thriller'].append(k)
        if ('Comic' in v) or ('Graphic' in v) or ('Manga' in v):
            genre_id_dict['Comics'].append(k)
        if ('Biography' in v) or ('Autobiography' in v) or ('Memoir' in v) or ('Sport' in v):
            genre_id_dict['Biography'].append(k)
        if ('Nonfiction' in v) or ('History' in v) or ('Politics' in v) or ('Cooking' in v) or ('Art' in v):
            genre_id_dict['Nonfiction'].append(k)
        if ('Self Help' in v) or ('Religion' in v):
            genre_id_dict['Self_help'].append(k)
        if ('Young Adult' in v):
            genre_id_dict['Young_Adult'].append(k)
        if ('Childrens' in v) or ('Family' in v):
            genre_id_dict['Family'].append(k)
        if ('Fiction' in v):
            genre_id_dict['Fiction'].append(k)        
    

    poss_books = genre_id_dict[genre]
    
    return filter_df(length, popularity, df.loc[poss_books])
    

# Cleaning functions

def lemmatize_stemming(text):
    '''
    Return lemmatized text
    '''
    word = WordNetLemmatizer().lemmatize(text, pos='v')
    return stemmer.stem(word)

def preprocess(text, stopwords_list):
    '''
    Returns text that is not longer than 3 chars or stop words (as defined by stopwords_list and gensim library) 
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords_list:
            result.append(lemmatize_stemming(token))
    return result

def item(id, ds):
    return ds.loc[ds['id'] == id]['titles'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(item_id, num, results, ds):
    item_id = ds.iloc[item_id]['id']
    print("Recommending " + str(num) + " products similar to " + item(item_id, ds) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print (rec[1])
        print("Recommended: " + item(rec[1], ds) + " (score:" + str(rec[0]) + ")")

    
    
    

# Rec system using only description

def get_recommendations(title, dff, sim_matrix, testing=False):
    '''
    Takes in a title and dataframe (use dff), then makes an abbreviated df containing only the titles and index number. 
    Returns top 10 similar books based on cosine similarity of vectorized description ALONE.
    '''
    if testing == True:
        title = find_title(title, dff)

    # create a new dataframe with titles as index, and index as a feature
    indices = pd.Series(list(range(len(dff))), index=dff.index)
    idx = indices[title]

    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    return dff.loc[indices[movie_indices].index][:11]




# helper functions for interfacing with the final recommendation system

def fail_to_find(df):
    final = input("That title did not match any of our books! Please try again, or enter 'quit!' to stop playing.")
    if final == 'quit!':
        return 0
    else:
        return find_title(final, df)
        
def find_title(guess, df):
    guess = guess.lower()
    final = []
    titles_list = {x.lower(): x for x in df.index}
    for possible in list(titles_list.keys()):
        if guess in possible:
               final.append(possible)
    if len(final) == 0:
        return fail_to_find(df)
    if len(final) == 1:
        print (f"\n Great! Looking for recomendations for the book: {titles_list[final[0]]}")
        return titles_list[final[0]]
    elif len(final) > 1:
        maybe = input(f"We found {len(final)} books that matched your search! Would you like to look thru them? If so enter'yes', otherwise enter 'no'.")
        if maybe == 'yes':
            print ("Is your book in this list? \n")
            maybe = input(f"{final}\n")
        for poss in final:
            end = input(f"Is your book {titles_list[poss]}? If so enter 'yes' and if not enter 'no'.")
            if end == 'yes':
                print (f"\n Great! Looking for recomendations for the book: {titles_list[poss]}")
                return titles_list[poss]
        return fail_to_find(df)
                     
                      
                      
                      
# Filter helper functions
                      
def return_pop_df(popularity, df):
    '''
    returns population filtered dataframe - options are:
        deep cut: < 27,000
        well known: between 80,000 and 27,000
        super popular: > 80,000
    '''
    if popularity == 'deep cut':
        return df[df['num_ratings'] < 27000]
    if popularity == 'well known':
        return df[(df['num_ratings'] < 80000) & (df['num_ratings'] > 27000)]
    if popularity == 'super popular':
        return df[df['num_ratings'] > 80000]
    
def filter_df(length, popularity, df):
    '''
    returns length and popularity filtered dataframe - 
    
    length options are:
        long: >= 350
        short: < 350
        
    popularity options are:
        deep cut: < 27,000
        well known: between 80,000 and 27,000
        super popular: > 80,000
    '''
    if length != None:
        if length == 'long':
            df = df[(df['pages'] >= 350)]
        elif length == 'short':
            df = df[(df['pages'] < 350)]
        
    if popularity != None:
        df = return_pop_df(popularity, df)

    return df



# Final recommendation system - includes option of filter arguments
                      
def recommendations(title, df, sim_matrix, filter_args=(None,None), list_length=11, suppress=True):
    '''
    Return recommendations based on a "bag of words" comprised of book author, genres and description.
    Function takes in title, list length, a dataframe, a similarity matrix and an option to add filters or suppress output.
    filter_args is (length, popularity) : 

    length options are:
        long: >= 350
        short: < 350
        
    popularity options are:
        deep cut: < 27,000
        well known: between 80,000 and 27,000
        super popular: > 80,000
    
    See filter_df() for further workings on this function.
    '''
    
    recommended_books = []
    
    # creating a Series for the movie titles so they are associated to an ordered numerical list
    indices = pd.DataFrame(df.titles, index=df.index)
    
    # getting the index of the book that matches the title
    idx = indices[indices.titles == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(sim_matrix[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:list_length+1].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_books.append(list(df.index)[i])
    
    if suppress == False:
        print (f"\n We recommend: ")
        for book_num in range(len(recommended_books)):
            print (book_num +1, recommended_books[book_num])
        
    if filter_args != (None,None):
        return filter_df(filter_args[0],filter_args[1],df.loc[recommended_books])

    return df.loc[recommended_books]
                      
