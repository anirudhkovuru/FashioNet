$(document).ready(function() {

                    $("#owl-demo").owlCarousel({
                        items :1,
                        itemsDesktop : [768,1],
                        itemsDesktopSmall : [414,1],
                        lazyLoad : true,
                        autoPlay : true,
                        navigation :true,
                        
                        navigationText :  false,
                        pagination : true,
                        
                    });
                      //owl//
                      $('.popup-with-zoom-anim').magnificPopup({
						type: 'inline',
						fixedContentPos: false,
						fixedBgPos: true,
						overflowY: 'auto',
						closeBtnInside: true,
						preloader: false,
						midClick: true,
						removalDelay: 300,
						mainClass: 'my-mfp-zoom-in'
					});
                     //magnificPopup//
			});