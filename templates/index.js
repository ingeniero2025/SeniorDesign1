function openModal(imgElement) {
    var modal = document.getElementById("myModal");
    var modalImg = document.getElementById("img01");
    modal.style.display = "block";
    modalImg.src = imgElement.src;
}

// Close the modal
function closeModal() {
    var modal = document.getElementById("myModal");
    modal.style.display = "none";
}