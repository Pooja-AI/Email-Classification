from data_Preprocessing import DataPreprocessing
from vector_back import Embedding
from models import model
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
data_df = pd.read_csv('Input_data.csv')
X = data_df.loc[:,'Input_text']
Y = data_df.loc[:,'Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=1, stratify=data_df['Category'])

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)

tstStr=['hi valeria check get 5k item short period time customer shortage help freundliche grüße best regard vasilios adamos rutronik elektronische bauelemente gmbh team leader purchasing industriestraße 2 de75228 ispringen phone 49 7231 8011728 fax 49 7231 8011633 email vasiliosadamosrutronikcom httpwwwrutronikcom httpwwwrutronikcom rutronik24 next generation ecommerce httpwwwrutronik24com httpwwwrutronik24com email including attachment may contain business trade secret confidential legally protected information received email error hereby notified review use copying distribution strictly prohibited please inform u immediately destroy email thank geschäftsführer helmut rudel thomas rudel markus krieg marco nabinger dr gregor sommer sitz der gesellschaft 75228 ispringen registergericht amtsgericht mannheim hrb 503663 printing please think environment bsp149 h6327 sp001058818 5k shortly availablevasiliosadamosrutronikcom']

tstStr2 = ['dear liza enclosed new order schedule please confirm within 2 working day kindly confirming cancellation push out thanks advance katalin katalin palmuller logistics buyer creating value increase customer competitiveness zalaegerszeg zrínyi út 38 h8900 36 92 50 7211 direct katalintothnepalmullerflexcom mailtokatalintothnepalmullerflexcom legal disclaimer information contained message may privileged confidential intended read individual entity addressed designee reader message intended recipient notice distribution message form strictly prohibited received message error please immediately notify sender delete destroy copy message infineon wk 512018katalin tothne palmuller']

count_test = count_vectorizer.transform(tstStr2)

loaded_model = pickle.load(open('nb.pkl', 'rb'))
result = loaded_model.predict(count_test)
print(result)

