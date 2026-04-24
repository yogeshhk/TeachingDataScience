import streamlit as st
from utils import (smiles_to_mol, mol_file_to_mol, 
                   draw_molecule, mol_to_tensor_graph, get_model_predictions)

# ----------- General things
st.title('HIV Inhibitor Dashboard')
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Model analysis"])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [DeepFindr](https://www.youtube.com/channel/UCScjF2g0_ZNy0Yv3KbsbR7Q)")
st.sidebar.image("assets/logo.png", width=100)

if page == "Predictor":
    # ----------- Inputs
    st.markdown("Select input molecule.")
    upload_columns = st.columns([2, 1])

    # File upload
    file_upload = upload_columns[0].expander(label="Upload a mol file")
    uploaded_file = file_upload.file_uploader("Choose a mol file", type=['mol'])

    # Smiles input
    smiles_select = upload_columns[0].expander(label="Specify SMILES string")
    smiles_string = smiles_select.text_input('Enter a valid SMILES string.')

    # If both are selected, give the option to swap between them
    if uploaded_file and smiles_string:
        selection = upload_columns[1].radio("Select input option", ["File", "SMILES"])

    if selection:
        if selection == "File":
            # Save it as temp file
            temp_filename = "temp.mol"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loaded_molecule = mol_file_to_mol(temp_filename)
        elif selection== "SMILES":
            loaded_molecule = smiles_to_mol(smiles_string)
    else:
        if uploaded_file:
            # Save it as temp file
            temp_filename = "temp.mol"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loaded_molecule = mol_file_to_mol(temp_filename)
        elif smiles_string:
            loaded_molecule = smiles_to_mol(smiles_string)

    # Set validity flag
    if loaded_molecule is None:
            valid_molecule = False
    else:
        valid_molecule = True

    # Draw if valid
    if not valid_molecule and (smiles_string != "" or uploaded_file is not None):
        st.error("This molecule appears to be invalid :no_entry_sign:")
    if valid_molecule and loaded_molecule is not None:
        st.info("This molecule appears to be valid :ballot_box_with_check:")
        pil_img = draw_molecule(loaded_molecule)
        upload_columns[1].image(pil_img)
        submit = upload_columns[1].button("Get predictions")

    # ----------- Submission
    st.markdown("""---""")
    if submit:
        with st.spinner(text="Fetching model prediction..."):
            # Convert molecule to graph features
            graph = mol_to_tensor_graph(loaded_molecule)
            # Call model endpoint
            prediction = get_model_predictions(graph)

        # ----------- Ouputs
        outputs = st.columns([2, 1])
        outputs[0].markdown("HIV Inhibitor Prediction: ")

        if prediction == 1:
            outputs[1].success("Yes")
        else:
            outputs[1].error("No")

        prediction_details = st.expander(label="Model details")
        details = prediction_details.columns([2, 1])

        # All of this is mocked
        details[0].markdown("Confidence: ")
        details[0].markdown("Model Version: ")
        details[0].markdown("Model Name: ")
        details[0].markdown("Test ROC: ")
        details[1].markdown("81%")
        details[1].markdown("1.0.1")
        details[1].markdown("Graph Transformer Network")
        details[1].markdown("0.84")
else:
    st.markdown("This page is not implemented yet :no_entry_sign:")




