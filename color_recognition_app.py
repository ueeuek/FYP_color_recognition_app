#https://github.com/whitphx/streamlit-webrtc
import streamlit as st
import cv2
import numpy as np
import joblib

import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import StringIO
from PIL import Image, ImageDraw

#https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
st.set_page_config(
    page_title="Color Recognition",
    page_icon="🥼",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.markdown(
    """
	<style>
	body {
		background-color: #AEC6CF;
		font-family: 'Courier', sans-serif;
		text-align: center;
	}
 	[data-testid="column"] {
		width: calc(25% - 1rem) !important;
		flex: 1 1 calc(25% - 1rem) !important;
		min-width: calc(20% - 1rem) !important;
	}
	.css-1l269bu {max-width:20% !important;}
	</style>
    """,
    unsafe_allow_html=True
)
# Using object notation
c1_sidebar, c2_sidebar = st.sidebar.columns([5, 1])
with c1_sidebar:
	st.markdown("<h1 style='text-align: right; font-family: Courier, sans-serif;'>Additional Info</h1>",
		unsafe_allow_html=True)
with c2_sidebar:
	st.write('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
	st.image(Image.open("images/about.png"), width=25)
	

st.sidebar.markdown("[basic guide on pairing outfits](https://www.stylecraze.com/articles/how-to-match-the-colors-of-your-clothes/)")
st.sidebar.markdown("[basic guide on pairing outfits - video](https://www.google.com/search?q=matching+colours+fashion&oq=matching+colours+fashion&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCDg4MzlqMGoxqAIAsAIA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:b3674b6a,vid:WLkVEvcuqT8)")
st.sidebar.markdown("[5 Different Looks](https://wonder-wardrobe.com/blog/5-color-outfit-matching-methods)")
st.sidebar.markdown("[psychology](https://www.verywellmind.com/color-psychology-2795824)")
st.sidebar.markdown("[Combinations](https://99designs.com/blog/tips/color-combinations/)")
st.sidebar.markdown("[Understand the colours](https://luxe.digital/lifestyle/style/color-matching-style-guide/)")
st.sidebar.markdown("[Guide on Formal Wear](https://pin.it/1daIPmu)")
st.sidebar.markdown("[(Men)Guide on Pairing Shoes and Pants](https://pin.it/6Zgqe5j)")
st.sidebar.markdown("[Guide on Bright Spring Colors](https://www.pinterest.com/pin/94012710963957801/)")

c1_header, c2_header = st.columns([5, 1])
with c1_header:		  
	st.markdown(
		"<h2 style='font-family: Courier, sans-serif;'>Color Recognition</h2>",
		unsafe_allow_html=True
	)
with c2_header:
#	st.write('\n\n\n')
	st.image(Image.open("images/header_shirt.png"))

#st.header('Color Recognition App 👕👖👗🛍')
#if st.button('Balloons?'):
#    st.balloons()

bytes_data = None
img_file_buffer = st.camera_input('Please provide an image for processing, ensure the image is taken under good lighting conditions for accurate processing.')
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	bytes_data = img_file_buffer.getvalue()
	st.empty()

elif uploaded_file is not None:
	bytes_data = uploaded_file.getvalue()
	st.empty()	

if bytes_data is None:
	st.stop()


knn = joblib.load('model.pkl')
###*********************************************************************************************************************************************************

img1 = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

st.markdown(
    """
    <div style='text-align: center; font-family: Courier, sans-serif;'>
        <h1>Colour Detected</h1>
        <p>Interact with the image to find your desired color!</p>
    </div>
    """,
    unsafe_allow_html=True
)

cv2.imwrite('ImageCaptured.jpg', img1)
#st.write(value)

def get_ellipse_coords(center):
	x, y = center
	radius = 10  # Adjust the radius according to your requirements
	coords = [(x - radius, y - radius), (x + radius, y + radius)]
	return coords

# Initialize the session state dictionary	
if "point" not in st.session_state:
	st.session_state["point"] = None
	st.session_state["value_text"] = None
	st.session_state["HSV_value_text"] = None
	st.session_state["colour_name_text"] = None
	st.session_state["suggestions_text"] = None

c1, c2 = st.columns(2)
with c1:
	with Image.open("ImageCaptured.jpg") as img:
		img.thumbnail((400, 400)) # Resize the image while maintaining the aspect ratio
		draw = ImageDraw.Draw(img)

		if st.session_state["point"] is not None:
			coords = get_ellipse_coords(st.session_state["point"])
			draw.ellipse(coords, fill="red")

		value = streamlit_image_coordinates(img, key="pil")

		if value is not None:
			point = value["x"], value["y"]

			if point != st.session_state["point"]:
				st.session_state["point"] = point
				#st.experimental_rerun()
				
				img_np = np.array(img)	
				HSVImg = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
				hsv = HSVImg[point[1], point[0]]

				HSVvalue = ''
				h = int(hsv[0]) / 180 * 360
				s = int(hsv[1]) / 255 * 100
				v = int(hsv[2]) / 255 *100
				HSVvalue = str(h) + ',' + str(s) + ',' + str(v)

				colour_prediction = knn.predict([[h,s,v]])

				## for matching colours
				matching_colours_dataset = pd.read_csv('suggestions.csv')

				# Search for combinations that contain the detected color
				suggestions = matching_colours_dataset[matching_colours_dataset['Color Combination'].str.contains(colour_prediction[0], case=False)]

				brightness = ''
				if colour_prediction[0] not in ['black', 'white']:
					Lightness_dict = {10: 'and very dark', 30: 'dark', 50: '', 70: 'light', 90: 'and very light'}
					Lightness_for_dict = min(Lightness_dict, key=lambda x: abs(x - v))
					brightness = Lightness_dict[Lightness_for_dict]
	
				# Store the state of st.text elements
				st.session_state['value_text'] = value if value is not None else ''
				st.session_state['HSV_value_text'] = HSVvalue if value is not None else ''
				st.session_state['brightness'] = brightness if value is not None else ''
				st.session_state['colour_name_text'] = colour_prediction[0] if value is not None else ''
				st.session_state['suggestions_text'] = suggestions if value is not None else ''
				st.experimental_rerun()

if st.session_state['HSV_value_text'] is None:
	text = 'Select any point on the image to know its color. You may tilt your phone to view full image !'
else:
	text = 'This is <span style="padding: 0px 6px"><strong>' + st.session_state['brightness'] + ' ' + st.session_state['colour_name_text'].upper() +\
		'</strong></span> <span style="display: inline-block; width: 30px; height: 18px; background-color:' + st.session_state['colour_name_text'] + '; margin-left: 6px;"></span><br>' +\
		'Suggest to match with: <br>'

	# Display st.text elements using stored state
	for index, row in st.session_state['suggestions_text'].iterrows():
		colors = row['Color Combination'].split(' and ')
		other_color = [color for color in colors if color != st.session_state['colour_name_text']][0]
		category = row['Category']
		text += f"<span style='display: inline-block; width: 30px; height: 18px; background-color:{other_color}; margin-left: 6px;'></span><span style='padding: 0px 6px'><strong>{other_color.upper()}</strong></span>: {category}<br>"

with c2:	
	st.markdown(
	f"<div style='font-family: Comic Sans MS, sans-serif; text-align: left; '>{text}</div>",
	unsafe_allow_html=True
    )
st.text_area('', '''⚠Please note that the suggestions provided are based on general associations and recommendations. The appearance of an outfit is influenced by multiple elements, including lighting conditions, personal preferences, and individual perception. Additionally, keep in mind that the shades and brightness of colors can greatly affect the overall look and feel of an outfit. We recommend considering these factors and experimenting with different combinations to find the perfect match for your style. Please do seek help or advice from the people around you for more accurate result!''')
