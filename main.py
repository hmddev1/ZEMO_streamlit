import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ZEMO import zemo
from zemo_code import calculate_zernike_moments, reconstruct_image
import io
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="ZEMO - Zernike Moments Calculator",
    # page_icon="📊",
    layout="wide"
)

with st.sidebar:
    st.title("ZEMO")
    st.markdown("""
    ### About
    **ZEMO** is a powerful image feature extraction tool powered by Zernike moments;
    ideal for analyzing astronomical, medical, and other visual data (even regular images) with 
    rotation-invariant precision.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Please upload an image with a size less than 2MB and a resolution of 200x200 pixels.")
    
    if uploaded_file is not None:
        order = st.number_input("Enter the Zernike order $P_{max}$:", 
                            min_value=1, 
                            max_value=50, 
                            value=1,
                            help="The Zernike order depends on the patterns, shapes, and details in the image. So, it is recommended to try $P_{max} = 15$ at the first try.")

# Main content
st.title("Feature Extraction by ZEMO")

if uploaded_file is not None:
    if uploaded_file.size > 2 * 1024 * 1024:
        st.error("File size should be less than 2MB")
    else:
        image = Image.open(uploaded_file)
        
        # Check if image is square
        if image.width != image.height:
            st.error("Please upload a square-shaped image (width = height)")
        else:
            # Resize image to 200x200
            image = image.resize((200, 200), Image.Resampling.LANCZOS)
            
            # Create two columns for images
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, 
                        caption='Original Image (200x200)', 
                        width=400)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                calculate_button = st.button("Calculate Zernike Moments", 
                                          use_container_width=True)
            
            if calculate_button:
                # Fixed image size of 200x200
                image_size = 200

                # Start timer
                start_time = time.time()
                
                st.write('Calculating... Please wait.')
                
                df, ZBFSTR, Z = calculate_zernike_moments(image, image_size=image_size, zernike_order=order)
                
                # Calculate duration
                duration = time.time() - start_time
                
                st.write(f'Done! Calculation took {duration:.2f} seconds.')
                st.write('Now we are reconstructing the image.')
                
                reconstructed = reconstruct_image(Z, image_size=image_size, ZBFSTR=ZBFSTR, show=False)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(reconstructed, cmap='bone', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'Reconstructed Image (Order = {order})')
                
                st.pyplot(fig)
                
                st.write("Zernike Moments:")
                st.write(df)
                
                st.write("Download Zernike Moments:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"ZMs_{uploaded_file.name}_order_{order}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # TXT Download
                    txt = df.to_csv(index=False, sep='\t').encode('utf-8')
                    st.download_button(
                        label="Download as TXT",
                        data=txt,
                        file_name=f"ZMs_{uploaded_file.name}_order_{order}.csv",
                        mime="text/plain",
                        use_container_width=True
                    )
else:
    st.info("Please upload an image using the sidebar to get started.")

# Add footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center'>
#     <p>Contact Information:</p>
#     <p>Email: your.email@example.com</p>
#     <p>Phone: +1234567890</p>
#     <p>Website: www.yourwebsite.com</p>
# </div>
# """, unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.markdown(
            '<h6>Made by <a href="https://github.com/hmddev1">@hmddev1</a></h6>',
            unsafe_allow_html=True,
        )
st.sidebar.markdown(
            '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/hmddev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="30" width="127"></a></div>',
            unsafe_allow_html=True,
        )