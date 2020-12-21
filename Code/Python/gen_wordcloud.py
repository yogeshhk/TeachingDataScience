# Python program to generate WordCloud 

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)


def data_from_text(txt_file):
    # comment_words = []
    text_so_far = ""
    with open(txt_file, 'r') as fin:
        for val in fin.readlines():
            tokens = [tok for tok in str(val).split() if tok.strip() != ""]
            if len(tokens) > 0:
                text_so_far += " ".join(tokens) + " "
                # comment_words.append(text_so_far)
    return [text_so_far]#comment_words


def data_from_csv(csv_file):
    df = pd.read_csv(csv_file, encoding="latin-1")
    comment_words = []
    for val in df.CONTENT:
        tokens = [tok for tok in str(val).split() if tok.strip() != ""]
        comment_words.append(" ".join(tokens) + " ")
    return comment_words


def gen_cloud(words_list):
    images = []
    for words in words_list:
        # print(words)
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(words)

        img = plt.imshow(wordcloud, animated=True)
        images.append([img])
    return images


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 8), facecolor=None)
    txt_filename = "data/ndtv.txt"
    if len(sys.argv) > 1:
        txt_filename = str(sys.argv[1])
    words_list = data_from_text(txt_filename)
    ims = gen_cloud(words_list)
    plt.axis("off")
    plt.tight_layout(pad=0)
    # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    # movie_file_name = txt_filename.replace("txt","mp4")
    image_file_name = txt_filename.replace("txt", "png")
    # ani.save(movie_file_name)
    fig.savefig(image_file_name)

    plt.show()
