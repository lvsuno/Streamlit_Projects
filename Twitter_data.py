import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pickle import load, dump
from PIL import Image
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import simplemma
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import  WordCloud, ImageColorGenerator
import advertools as adv
import preprocess_lvsuno as pl
import pydeck as pdk
import os
import re

# Title
st.title('Some twitter data between 24 February 2022 and 25 February 2022')
st.text("We keep only the data where we\'ve a geographic coordinates (explicit or in A bounding box)")
st.markdown('**The data used can be found on** (https://archive.org/search?query=twitterstream&page=3&sort=-publicdate)')


# Global value
DATE_COLUMN = 'created_at'
DATA_PATHS_1 = ('data/20220224/Geo_20220224_full.json')
DATA_PATHS_2 = ('data/20220225/Geo_20220225_full.json')
COUNTRY_PATHS = ('data/Country_average_latlong.csv')
MAPBOX_API_KEY = os.environ["MAPBOX_API_KEY"]

# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Preprocessing
def prepro(text):
    # Put tweets into the lower case
    text = text.lower()
    # Expand any contracted words
    text = pl.cont_exp(text)

    # remove emails
    text = pl.remove_emails(text)
    # remove urls
    text = pl.remove_urls(text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # remove retweets
    text = pl.remove_rt(text)
    # removal html tags
    text = pl.remove_html_tags(text)
    # remove special characters
    text = pl.remove_special_chars(text)
    # Remove multiple space characters
    text = re.sub('\s+', ' ', text)
    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)
    # remove successive characters
    text = re.sub("(.)\\1{2,}", "\\1", text)
    return text

# Load the data
@st.cache_data
def load_data(choose_date):
    if choose_date == '24':
       data = pd.read_json(DATA_PATHS_1) #,  nrows=nrows)
       data['content_pre'] = data['content'].apply(prepro)
       #lowercase = lambda x: str(x).lower()
       #data.rename(lowercase, axis='columns', inplace=True)
       data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
       data = data.sort_values('date_hour', ascending=True)
    elif choose_date == '25':
       data = pd.read_json(DATA_PATHS_2) #,  nrows=nrows)
       data['content_pre'] = data['content'].apply(prepro)
       #lowercase = lambda x: str(x).lower()
       #data.rename(lowercase, axis='columns', inplace=True)
       data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
       data = data.sort_values('date_hour', ascending=True)
    else:
       dt = pd.read_json(DATA_PATHS_1) #,  nrows=nrows)
       dt1 = pd.read_json(DATA_PATHS_2) #,  nrows=nrows)
       #lowercase = lambda x: str(x).lower()
       #dt.rename(lowercase, axis='columns', inplace=True)
       #dt1.rename(lowercase, axis='columns', inplace=True)
       dt[DATE_COLUMN] = pd.to_datetime(dt[DATE_COLUMN])
       dt1[DATE_COLUMN] = pd.to_datetime(dt1[DATE_COLUMN])
       data = pd.concat([dt, dt1], ignore_index=True)
       data['content_pre'] = data['content'].apply(prepro)
       data = data.sample(frac=1).reset_index(drop=True)
       data = data.sort_values('date_hour', ascending=True)

    return data #.sample(10000)

# Find and save emoji and stats
def find_and_save_emoji(dat, date):
    """
    Get the mentions in a set of twitter's text
    :param dat: data series column
    :return: list of mentions
    """
    dat = dat.tolist()
    emoji_summary = adv.extract_emoji(dat)
    emoji_list = emoji_summary['top_emoji']
    dump(emoji_summary, open('saved_emoji/emoji_summary_'+date+'.pkl', 'wb'))
   

@st.cache_data
def plot_emoji(date, n=20, order='top', data=None):
    if not os.path.isfile('saved_emoji/emoji_summary_'+date+'.pkl'):
        if not os.path.isdir('saved_emoji'):
            os.mkdir('saved_emoji')
        find_and_save_emoji(data, date)

    emoji_summary = load(open('saved_emoji/emoji_summary_'+date+'.pkl', 'rb'))
    emoji_list = emoji_summary['top_emoji']
    if order == 'last':
        labels = list(list(zip(*emoji_list[-n:]))[0])
        count = list(list(zip(*emoji_list[-n:]))[1])
    else:
        labels = list(list(zip(*emoji_list[0:n]))[0])
        count = list(list(zip(*emoji_list[0:n]))[1])
    fig = go.Figure([go.Bar(x=labels, y=count)])
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")
    


@st.cache_data
def get_words_lem(dat, lang):
    try:
        dt = dat['content_pre'].apply(lambda x: simplemma.lemmatize(x, lang=lang))
    except Exception:
        Wordlem = WordNetLemmatizer()
        dt = dat['content_pre'].apply(Wordlem.lemmatize)
    words_lem = ' '.join([word for word in dt])
    return words_lem


# Get Top n Big N-grams
def get_top_n_Ngram(corpus, stpw, n=None, ngram_range=(1, 1)):
    """
    Take in entry the following parameters
    :param verbose: print or not the table of the word and their frequency
    :param corpus: The corpus
    :param n: The number of top Ngrams
    :param ngram_range: a tuple of ngram (1, 2) means one to 2 gram
    :param stpw: language of stopword or list of stopwords
    :return: word frequency
    """
    Vect = CountVectorizer(ngram_range=ngram_range, stop_words=stpw).fit(corpus)
    Bag_of_words = Vect.transform(corpus)
    Sum_words = Bag_of_words.sum(axis=0)
    Words_freq = [(word, Sum_words[0, idx]) for word, idx in Vect.vocabulary_.items()]
    Words_freq = sorted(Words_freq, key=lambda x: x[1], reverse=True)
    common_words = Words_freq[:n]
    df1 = pd.DataFrame(common_words, columns=['NGram', 'Count'])
    df1 = df1.groupby('NGram').sum()['Count'].sort_values(ascending=False)

    return df1.reset_index()


# Function to create wordcloud image giving a list of words
@st.cache_data
def create_word_cloud(word, mask_image, stop_w, dest_file, col=True):
    """

    :param word: list of word
    :param mask_image: path to the mask
    :param stop_w: list of stopword
    :param dest_file: destination file
    :param col: collocations is true of false
    :return:
    """
    if not os.path.isfile(dest_file):
        mask = np.array(Image.open(mask_image).convert("RGB"))
        mask[mask.sum(axis=2) == 0] = 255
        if stop_w is None:
            stpwd = stop_w
        else:
            stpwd = set(stop_w)

        word_cloud = WordCloud(contour_width=2, 
                               background_color="white",
                               # mode="RGBA",
                               stopwords=stpwd,
                               mask=mask,
                               collocations=col,
                               color_func=ImageColorGenerator(mask)).generate(word)
        # Create coloring from the image
        fig = plt.figure(figsize=(20,20))
        plt.axis('off')
        plt.tight_layout(pad=0)
        # word_cloud.recolor(color_func=img_colors)
        plt.imshow(word_cloud, interpolation="bilinear")

        # Store the image
        plt.savefig(dest_file, format="png")
        st.pyplot(fig)
    else:
        image = Image.open(dest_file)
        st.image(image, use_column_width='always')




def get_mentions_list(dat):
    """
    Get the mentions in a set of twitter's text
    :param dat: data frame column
    :return: list of mentions
    """
    # Retrieves all occurrences of @+text
    dat = dat.astype(str)
    col = dat.columns
    dat[col[0]] = dat[col[0]].str.findall(r'@\w+')
    # dat1 = dat.apply(lambda x: re.findall(r'@\w+', x))
    # Removes the @ in front
    dat[col[0]] = [list(map(lambda x: x[1:], mentions)) for mentions in dat[col[0]]]
    # Converts the list of words in each row to a string
    dat[col[0]] = dat[col[0]].apply(lambda x: ' '.join(x))
    # Concatenates all strings in one string
    all_mentions = ' '.join([word for word in dat[col[0]]])
    return all_mentions



def get_hashtag_list(dat):
    """
    Get the mentions in a set of twitter's text
    :param dat: data frame column
    :return: list of mentions
    """
    # Retrieves all occurrences of #+text
    dat = dat.astype(str)
    col = dat.columns
    dat[col[0]] = dat[col[0]].str.findall(r'#\w+')
    # dat1 = dat.apply(lambda x: re.findall(r'@\w+', x))
    # Removes the @ in front
    dat[col[0]] = [list(map(lambda x: x[1:], mentions)) for mentions in dat[col[0]]]
    # Converts the list of words in each row to a string
    dat[col[0]] = dat[col[0]].apply(lambda x: ' '.join(x))
    # Concatenates all strings in one string
    all_hashtags = ' '.join([word for word in dat[col[0]]])
    return all_hashtags

@st.cache_data
def get_twitts_count_per_country(dt, co_paths):
    co = pd.read_csv(co_paths, header=0, keep_default_na=False)
    sf = dt['Country'].value_counts()
    sf = pd.DataFrame({'Country':sf.index, 'n_twitts':sf.values})
    sf = sf.join(co.set_index('Country'), on='Country')
    sf= sf[sf['lat'].notnull() | sf['lon'].notnull()]

    sf1=dt.loc[(dt['Country']=='None') | (dt['Country']==''),["lat", "lon"]]
    sf1.insert(0,"Country",np.full(sf1.shape[0],None),True)
    sf1.insert(1,"n_twitts",np.ones(sf1.shape[0]),True)
    sf1.insert(2,"ISO 3166 Country Code",np.full(sf1.shape[0],None),True)
    sf = pd.concat([sf, sf1], ignore_index=True)
    return sf



# Sidebar two choose the date
with st.sidebar:
    Menu_radio = st.radio(
        "Go to",
        ("24 February", "25 February", "Both")
    )
       
    "---"

# Radio Button to choose between the day

date_radio = st.radio(
        "Choose a day",
        ("24 February", "25 February", "Both"),
        horizontal=True
    )
if date_radio == '24 February':
    data_load_state = st.text('Loading data...')
    data = load_data('24')
    data_load_state.text("Done!")
elif date_radio == '25 February':
    data_load_state = st.text('Loading data...')
    data = load_data('25')
    data_load_state.text("Done!")
else:
    data_load_state = st.text('Loading data...')
    data = load_data('both')
    data_load_state.text("Done!")


# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache_data)")

tab1, tab2, tab3, tab4 = st.tabs(["🗃 Data", "📈 Chart", "🗺️ Map", "👨‍💻 Tweets analysis"])

with tab1:
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        dt = data if data.shape[0]<= 10000 else data.sample(10000)
        st.write(dt)
        st.write(get_twitts_count_per_country(data, COUNTRY_PATHS))

with tab2:
    # Number of tweets by hour
    st.subheader('Number of tweets by hour')
    hist_hour = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_hour)
    # Number of tweets by language
    st.subheader('Number of tweets by language')
    # fig, ax = plt.subplots()
    hist_lang = data['lang'].value_counts()
    hist_lang = hist_lang.sort_values(ascending=True)
    st.bar_chart(hist_lang)
    # Bar plot of the top 20 emojis used
    st.subheader('Bar plot of the top 20 emojis used')
    plot_emoji(date_radio, 20, 'top', data['content'])
    # Define a selectbox and plot the 20 most frequent bi-grams
    lang_selectbox_ngram = st.selectbox('Select a language',('english', 'french'))
    Most_used_words_2 = get_top_n_Ngram(data.loc[data['lang']==lang_selectbox_ngram,'content_pre'],list(adv.stopwords[lang_selectbox_ngram]) , n=20,ngram_range=(2, 2))
    fig = px.bar(Most_used_words_2, x='NGram',y='Count',title='The 20 most frequent bi-grams in the dataset')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    fig.write_image('images/Most_used_words_2_'+date_radio+'_'+lang_selectbox_ngram+'.png')
    # Define a selectbox and plot the 20 most frequent tri-grams
    Most_used_words_3 = get_top_n_Ngram(data.loc[data['lang']==lang_selectbox_ngram,'content_pre'],list(adv.stopwords[lang_selectbox_ngram]), n=20 ,ngram_range=(3, 3))
    fig1 = px.bar(Most_used_words_3, x='NGram',y='Count',title='The 20 most frequent tri-grams in the dataset')
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

    fig1.write_image('images/Most_used_words_3_'+date_radio+'_'+lang_selectbox_ngram+'.png')

    

with tab3:
    # Some number in the range 0-23
    # map_col1, map_col2= st.columns(2)
    #with map_col1:
         # hour_to_filter = st.slider('hour', 0, 23, 18)
         # filtered_data = data[data['date_hour'] == hour_to_filter]
         # st.subheader('Map of all Twitts at %s:00' % hour_to_filter)
    # fig = px.scatter_mapbox(filtered_data, lat="lat", lon="lon",hover_name= "content", zoom=0)
    # fig.update_traces(cluster=dict(enabled=True))

    #with map_col2:
    lang_checkbox = st.checkbox('Display by language')
    px.set_mapbox_access_token(open(".mapbox_token").read())
    if lang_checkbox:
        fig = px.scatter_geo(data,lat="lat", lon="lon", hover_name= "content", color="lang", 
                              animation_frame="date_hour")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    elif ~lang_checkbox:
        fig = px.scatter_geo(data,lat="lat", lon="lon", hover_name= "content", color="Country", 
                              animation_frame="date_hour")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    # st.map(filtered_data)
    co_nb_twi = get_twitts_count_per_country(data, COUNTRY_PATHS)

    st.markdown('# Plot the number of tweets by known country.')


    view = pdk.data_utils.compute_view(co_nb_twi[['lon', 'lat']])
    view.pitch= 40
    #view.bearing=-27.36
    
    Scatter_layer = pdk.Layer('ScatterplotLayer',
                         data=co_nb_twi,
                         get_position= '[lon, lat]',
                         get_radius="n_twitts * 1000",
                         radius_min_pixels=1,
                         radius_max_pixels=10,
                         line_width_min_pixels=1,
                         get_fill_color=[0, 'lon > 0 ? 200 * lon : -200 * lon', 'lon', 140],
                         pickable=True,
                         auto_highlight=True)
    Column_layer = pdk.Layer('ColumnLayer',
                             data=co_nb_twi,
                             get_position='[lon, lat]',
                             radius=100000,
                             elevation_scale=1000,
                             uto_highlight=True,
                             get_elevation="n_twitts",
                             get_fill_color=[255, 'n_twitts >255 ? n_twitts/2000 : n_twitts', 'n_twitts/255', 140],
                             pickable=True,
                             extruded=True)
    tooltip = {"html": "<b>Number of twitts:</b> {n_twitts}<br/>"
               "<b>Country:</b> {Country}",
                "style": {"backgroundColor": "steelblue",
                 "color": "white"}}
    st.pydeck_chart(pdk.Deck(
        layers= [Scatter_layer, Column_layer],
        map_provider="mapbox",
        map_style=pdk.map_styles.SATELLITE,
        initial_view_state=view,
        tooltip=tooltip
        ))
    
with tab4:
    st.markdown("## Twitter's mentions")
    create_word_cloud(get_mentions_list(data['content'].to_frame()), "images/twitter_logo.png", None,
                  "images/Mentions_twitter_"+date_radio+".png", col=False)
    
    st.markdown("## Twitter's hashtags")
    create_word_cloud(get_hashtag_list(data['content'].to_frame()), "images/twitter_logo.png", None,
                  "images/Hashtag_twitter_"+date_radio+".png", col=False)
    st.markdown("## Twitter's Word Count")
    data_1 = data[data['lang']!='unknown']
    lang = data_1['lang'].unique().tolist()
    lang_selectbox = st.selectbox('Select a language',lang)
    if (lang_selectbox not in ['arabic', 'chinease', 'indonesian', 'japanese', 'italian',
                                'hindi', 'haitian','vietnamese','albanian','thai',
                                'korean', 'welsh', 'romanian', 'central khmer','nepali',
                                'tamil','urdu','marathi','somali','sinhala','persian',
                                'malayalam','central kurdish (sorani)','dhivehi','kannada',
                                'lao','sindhi','panjabi','oriya (macrolanguage)','pushto',
                                'hebrew','gujarati','bengali','telugu',]):
        words_lem = get_words_lem(data[data['lang']==lang_selectbox],lang_selectbox)
        try:
            STOPWORDS = adv.stopwords[lang_selectbox]
        except Exception:
            try: 
                STOPWORDS = stopwords.words(lang_selectbox)
            except Exception:
                STOPWORDS = None
        create_word_cloud(words_lem, "images/twitter_logo.png", STOPWORDS, "images/Words_twitter_"+date_radio+"_"+lang_selectbox+"_.png")
    else:
        st.warning('This language is not supported by our system')




