import com.pedropathing.math.Vector2D;
import com.pedropathing.paths.curves.bezier.BezierCurve;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PedroProbe {
    public static void main(String[] args) throws Exception {
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String line;
        int count = 0;
        while ((line = input.readLine()) != null) {
            if (++count > 10100) throw new IllegalArgumentException("Case limit exceeded");
            String[] fields = line.trim().split("\\s+");
            if (fields.length != 10) throw new IllegalArgumentException("Expected id, t and eight coordinates");
            int id = Integer.parseInt(fields[0]);
            double parameter = Double.parseDouble(fields[1]);
            Vector2D[] points = new Vector2D[4];
            for (int index = 0; index < points.length; index++) {
                points[index] = Vector2D.cartesian(
                        Double.parseDouble(fields[2 + index * 2]), Double.parseDouble(fields[3 + index * 2]));
            }
            BezierCurve curve = new BezierCurve(points);
            Vector2D point = curve.get(parameter);
            Vector2D first = curve.derivative(parameter);
            Vector2D second = curve.getDerivative(2, parameter);
            System.out.println(id + " " + point.x() + " " + point.y() + " "
                    + first.x() + " " + first.y() + " " + second.x() + " " + second.y() + " "
                    + curve.curvature(parameter));
        }
    }
}
