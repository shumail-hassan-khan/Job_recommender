#create virtual environment first 
python -m venv venv

#activate virtual environment
Source venv/Scripts/activate

#install dependencies
pip install -r requirements.txt


#go to project path 
cd app

#run your project 
uvicorn main:app --reload --port 7777

#project will run on this port 
http://localhost:7777/


