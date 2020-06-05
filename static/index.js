(function () {
    var canvas = document.querySelector("#canvas");
    var context = canvas.getContext("2d");
    canvas.width = 280;
    canvas.height = 280;

    var mouse = {x:0, y:0};
    var lastMouse = {x:0, y:0};
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.color = "black";
    context.lineWidth = 15;
    context.lineJoin = context.lineCap = "round";

    debug();

    canvas.addEventListener(
        "mousemove",
        function(e) {
            lastMouse.x = mouse.x;
            lastMouse.y = mouse.y;
            mouse.x = e.pageX - this.offsetLeft;
            mouse.y = e.pageY - this.offsetTop;
        },
        false
    );

    canvas.addEventListener(
        "mousedown",
        function () {
            canvas.addEventListener("mousemove", onPaint, false);
        },
        false
    );

    canvas.addEventListener(
        "mouseup",
        function () {
            canvas.removeEventListener("mousemove", onPaint, false);
        },
        false
    );

    var onPaint = function () {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(mouse.x, mouse.y);
        context.closePath();

        context.stroke();
    };

    function debug() {
        var clearBtn = $("#clearBtn");
        clearBtn.on(
            "click", 
            function () {
                context.clearRect(0, 0, 280, 280);
                context.fillStyle="white";
                context.fillRect(0, 0, canvas.width, canvas.height);
            }
        );

        // context.lineWidth = $(this).val();
        context.lineWidth = 14;

        $("#lineWidth").change(
            function () {
                context.lineWidth = $(this).val();
            }
        )
    }
}());
