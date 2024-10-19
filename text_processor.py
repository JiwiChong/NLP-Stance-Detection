import re
import nltk 
from nltk.corpus import stopwords

nltk.download('stopwords')
sw = stopwords.words('english')
sw.append('#semst')

def clean_text(txt):

    if type(txt) != str:
        return ''

    text = ' '.join(re.sub("([@][A-Za-z0-9_]+)|(#SemST)|(\w+:\/\/\S+)"," ", txt).split())
    text = text.lower()
    rx = re.compile('([\[\"\'`´&#/\\,./\-+()_$!?%*\n\t\r\f1234567890:;<>\]])')
    text = rx.sub(' ', text)
    # remove single letter

    text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()

    stopwords_regex = re.compile(r'\b(' + r'|'.join(sw) + r')\b\s*')
    text = stopwords_regex.sub(' ', text)
    return re.sub('\s+',' ',re.sub('\'',' ',text)).strip()#.split(' ')