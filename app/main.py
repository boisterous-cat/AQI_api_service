from fastapi import FastAPI, UploadFile, WebSocket, Request
from io import BytesIO
from fastapi.responses import FileResponse, Response, PlainTextResponse
from requests_cache import RedisCache
from retry_requests import retry
import requests_cache
import openmeteo_requests
import pickle
import gzip
from typing import Any
from fastapi.encoders import jsonable_encoder
from data_request_model import *
import pandas as pd
from geopy.geocoders import Nominatim

from pathlib import Path
from fastapi.templating import Jinja2Templates
import asyncio
import logging
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from time import sleep
from random import randint
from config import REDIS_HOST, REDIS_PORT

base_dir = Path(__file__).resolve().parent

with gzip.open('Data/pickle_model.pkl', 'rb') as ifp:
    MODEL = pickle.load(ifp)

app = FastAPI(title="Year project - AQI API", description="API to predict air quality index")

templates = Jinja2Templates(directory=str(Path(base_dir, "templates")))
LOGFILE = "app.log"

# create logger and configure
logger = logging.getLogger("log_app")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOGFILE)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('API is starting up')


@app.get("/")
def read_root():
    return FileResponse('body.html')


@app.post("/predict", summary='Get one prediction')
def predict(parameters: MlRequest) -> float:
    """
        Получаем предсказание индекса качества воздуха по одному объекту.
        Args:
            parameters: объект класса MlRequest

        Returns:
            float: индекс качества воздуха.
    """
    result = 0.0
    try:
        predicted = predict_items([parameters])
        result = predicted[0] if len(predicted) else None
    except Exception as e:
        logger.error(f"Ошибка при предсказании по объекту {e}")
    return result


@app.post("/predict_items", summary='Get predictions for several items')
def predict_items(items: List[MlRequest]) -> List[float]:
    """
            Получаем json, создаем из него датафрейм и отправляем за предсказаниями.

            Args:
                items: список объектов MLRequests в формате json

            Returns:
                List[float]: список предсказанных индексов.
    """
    results = []
    df = pd.DataFrame(jsonable_encoder(items))
    if len(df) != 0:
        # Rename the columns for clarity
        df.columns = ['CO AQI Value', 'NO2 AQI Value', 'Ozone AQI Value', 'PM2.5 AQI Value']
        results = [float(i) for i in predict_by_df(df)]
    return results


@app.post("/csv_content", summary='Get predictions for csv')
def get_csv(file: UploadFile) -> Response:
    """
                Делаем предсказания по файлу в формате csv.

                Args:
                    file: файл csv
    """
    df = pd.DataFrame()
    try:
        content = file.file.read()
        buffer = BytesIO(content)
        df = pd.read_csv(buffer, index_col=False)
        buffer.close()
        file.close()
    except Exception as e:
        logger.error(f"Проблема с чтением файла {e}")
    if not df.empty:
        try:
            df['predict'] = pd.Series(predict_by_df(df))
            df.to_csv('predictions.csv')
            return FileResponse(path='predictions.csv', media_type='text/csv', filename='predictions.csv')
        except Exception as e:
            logger.error(f"Проблема с возвратом файла {e}")
            return PlainTextResponse("No GraphQL query found in the request", 400)


@app.post("/get_data", summary='Get predictions for location')
def predict_by_coordinates(location: UserLocation):
    """
    По наименованию страны и города получаем координаты (сервис geocode) и
    при помощи открытого апи получаем показатели на текущий момент. Затем отправляем их в модель.
    Args:
        location: страна и город указанные пользователем
    Returns:
         float: предсказание
    """

    # calling the Nominatim tool
    loc = Nominatim(user_agent="GetLoc")
    user_location = location.City + " " + location.Country
    # entering the location name
    get_loc = __reverse_geocode(loc, user_location)

    if get_loc is None:
        return PlainTextResponse("Упс! Что-то пошло не так, проверьте данные или попробуйте позднее.")
    else:
        return __get_data_from_open_meteo(get_loc.latitude, get_loc.longitude)


def __reverse_geocode(geolocator, loc, sleep_sec=0):
    try:
        return geolocator.geocode(loc)
    except GeocoderTimedOut:
        logger.info('TIMED OUT: GeocoderTimedOut: Retrying...')
        sleep(randint(1*100, sleep_sec*100)/100)
        return __reverse_geocode(geolocator, loc, sleep_sec)
    except GeocoderServiceError as e:
        logger.info('CONNECTION REFUSED: GeocoderServiceError encountered.')
        logger.error(e)
        return None
    except Exception as e:
        logger.info('ERROR: Terminating due to exception {}'.format(e))
        return None


def __get_data_from_open_meteo(latitude: float, longitude: float) -> float:
    # Setup the Open-Meteo API client with cache and retry on error
    backend = RedisCache(host=REDIS_HOST, port=REDIS_PORT)
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600, backend=backend)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    predicted = predict_items([__convert_measure(response)])
    return predicted[0] if len(predicted) else None


def predict_by_df(df):

    predictions = MODEL.predict(df.dropna())
    return predictions


def __convert_measure(response: Any) -> MlRequest:
    # расчеты
    # https://sciencing.com/convert-ppm-micrograms-per-cubic-meter-2727.html#:~:text=Converting%20From%20Microgram%20per%20Meter%20Cube%20to%20PPM
    current = response.Current()
    pm2 = current.Variables(0).Value()
    ozone = current.Variables(3).Value()/1000
    co = current.Variables(1).Value()/1000
    no = (current.Variables(2).Value() * 24.24) / 46.01

    item = MlRequest(pm2=pm2, ozone=ozone, co=co, no=no)
    return item


async def log_reader(n=5):
    """Log reader

        Args:
            n (int, optional): number of lines to read from file. Defaults to 5.

        Returns:
            list: List containing last n-lines in log file with html tags.
    """
    log_lines = []
    with open(f"{LOGFILE}", "r") as file:
        for line in file.readlines()[-n:]:
            if line.__contains__("ERROR"):
                log_lines.append(f'<span class="text-red-400">{line}</span><br/>')
            elif line.__contains__("WARNING"):
                log_lines.append(f'<span class="text-orange-300">{line}</span><br/>')
            else:
                log_lines.append(f"{line}<br/>")
        return log_lines


# only localhost
@app.websocket("/ws/log")
async def websocket_endpoint_log(websocket: WebSocket):
    """WebSocket endpoint for client connections

        Args:
            websocket (WebSocket): WebSocket request from client.
    """
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(1)
            logs = await log_reader(30)
            await websocket.send_text(logs)
    except Exception as e:
        print(e)
    finally:
        await websocket.close()


# only localhost
@app.get("/logs")
async def get(request: Request):
    """Log file viewer

        Args:
            request (Request): Default web request.

        Returns:
            TemplateResponse: Jinja template with context data.
        """
    context = {"title": "FastAPI Streaming Log Viewer over WebSockets", "log_file": LOGFILE}
    return templates.TemplateResponse("index.html", {"request": request, "context": context})
