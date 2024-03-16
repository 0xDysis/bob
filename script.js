const gallery = document.querySelector('.gallery');
const prevBtn = document.querySelector('.prev-btn');
const nextBtn = document.querySelector('.next-btn');
const galleryItems = document.querySelectorAll('.gallery ul li');
const itemWidth = galleryItems[0].offsetWidth;

let currentIndex = 0;

prevBtn.addEventListener('click', () => {
  currentIndex--;
  if (currentIndex < 0) {
    currentIndex = galleryItems.length - 1;
  }
  scrollToItem(currentIndex);
});

nextBtn.addEventListener('click', () => {
  currentIndex++;
  if (currentIndex >= galleryItems.length) {
    currentIndex = 0;
  }
  scrollToItem(currentIndex);
});

function scrollToItem(index) {
  const scrollPosition = index * itemWidth;
  gallery.scrollTo({
    left: scrollPosition,
    behavior: 'smooth'
  });
}