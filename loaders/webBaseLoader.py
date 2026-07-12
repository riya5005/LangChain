from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model=ChatOllama(
    model="llama3.2",
    temperature=0.7)


prompt=PromptTemplate(
    template="answer the following questions \n{question} from the following text \n{text}",
    input_variables=['question','text']
)
parser=StrOutputParser()
url='https://www.flipkart.com/samsung-galaxy-book4-metal-intel-core-i7-13th-gen-1355u-16-gb-512-gb-ssd-windows-11-home-np750xgj-kg3in-np750xgj-lg3in-np750xgj-lg8in-thin-light-laptop/p/itm1df5212efb9fc?pid=COMGZUYDYPUZAAXD&lid=LSTCOMGZUYDYPUZAAXDBMAOEF&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=en_FNFqwucDslh4F-KWYyXk5Yb1u3Vz9AN_FEpcLbWwUb_VRb-ukPBS1ntXQgKd6JEn4vT2z1ndc-TSRKJByvWw2yY_OPhUuh8UIFKrYdeIm-7DR9Pbc0f5qJty02NPbkLH&ppt=None&ppn=None&ssid=fdxeo49gao0000001783861394697&ov_redirect=true'
loader=WebBaseLoader(url)
docs=loader.load()
chain=prompt|model|parser
result=chain.invoke({'question':"what is the peak price of this product",'text':docs[0].page_content})
print(result)